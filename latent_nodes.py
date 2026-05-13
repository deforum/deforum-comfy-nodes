import torch
import comfy.samplers
import comfy.model_management
from .tools import VariantSupport
from .base_node import LatentNode, NODE_POSTFIX

# -----------------------------------------------------------------------------
#  Helper: SLERP, LERP, NLERP between tensors
# -----------------------------------------------------------------------------

def _lerp(t: float, a: torch.Tensor, b: torch.Tensor):
    return a + t * (b - a)

def _slerp(t: float, a: torch.Tensor, b: torch.Tensor):
    """Spherical interpolation along surface of hypersphere."""
    dots = (a.flatten(1) * b.flatten(1)).sum(-1)
    omega = torch.acos(torch.clamp(dots, -1 + 1e-7, 1 - 1e-7))
    so = torch.sin(omega)
    res = (torch.sin((1 - t) * omega) / so).unsqueeze(-1) * a.flatten(1) + (
        torch.sin(t * omega) / so
    ).unsqueeze(-1) * b.flatten(1)
    return res.reshape_as(a)

def _nlerp(t: float, a: torch.Tensor, b: torch.Tensor):
    lerped = _lerp(t, a, b)
    norm = torch.norm(lerped.flatten(1), dim=-1, keepdim=True)
    norm[norm == 0] = 1
    return (lerped.flatten(1) / norm).reshape_as(a)

_INTERP_FUNCS = {"slerp": _slerp, "lerp": _lerp, "nlerp": _nlerp}

# -----------------------------------------------------------------------------
#  Node 1: Seed‑Interpolated Noise (ε batch generator)
# -----------------------------------------------------------------------------

MAX_RESOLUTION=32768

# from  https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
def slerp(val, low, high):
    dims = low.shape

    #flatten to batches
    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)

    # in case we divide by zero
    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0

    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res.reshape(dims)

@VariantSupport()
class SeedInterpNoise(LatentNode):
    """
    Produce a batch of noise tensors whose seeds advance by +1 after
    every (interp_steps + 1) frames, with SLERP interpolation between
    the two anchor seeds.

    Example (start_seed=1, interp_steps=2, frames=7):
        seeds: 1   1--2   1--2   2   2--3   2--3   3
                ε0  ε1    ε2    ε3  ε4     ε5     ε6
                ▲   ▲          ▲           ▲
                |   |__________|           |
                |______ segment 1 _____|   |___ seg 2 …
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "source": (["CPU", "GPU"],),
                    "start_seed": ("INT",  {"default": 0, "min": 0,
                                            "max": 0xffffffffffffffff}),
                    "frames":     ("INT",  {"default": 8, "min": 1,
                                            "max": 9999999}),
                    "interp_steps": ("INT", {"default": 1, "min": 0,
                                             "max": 1024}),
                    "width":  ("INT", {"default": 512, "min": 64,
                                       "max": MAX_RESOLUTION, "step": 8}),
                    "height": ("INT", {"default": 512, "min": 64,
                                       "max": MAX_RESOLUTION, "step": 8}),
               }}

    RETURN_TYPES = ("LATENT",)          # matches NoisyLatentImage
    FUNCTION = "build"

    def build(self, source, start_seed, frames, interp_steps,
              width, height):
        device = "cpu" if source == "CPU" \
                 else comfy.model_management.get_torch_device()
        c, h, w = 4, height // 8, width // 8

        # pre-allocate output tensor
        batch = torch.empty((frames, c, h, w),
                            dtype=torch.float32, device=device)

        # helper: epsilon(seed)
        def eps(seed):
            g = torch.Generator(device).manual_seed(seed)
            return torch.randn((c, h, w), generator=g, device=device)

        seg_len = interp_steps + 1      # ε_a + k in-betweens + ε_b
        cur_seed = start_seed
        frame_idx = 0

        while frame_idx < frames:
            eps_a = eps(cur_seed)
            eps_b = eps(cur_seed + 1)

            # anchor A
            batch[frame_idx] = eps_a
            frame_idx += 1
            if frame_idx >= frames:
                break

            # interpolated frames
            for k in range(1, seg_len):
                if frame_idx >= frames:
                    break
                t = k / seg_len
                batch[frame_idx] = slerp(t, eps_a, eps_b)
                frame_idx += 1

            # anchor B (start of next segment) only if we still need frames
            cur_seed += 1

        return ({"samples": batch.cpu()}, )

_VALID_SAMPLERS = comfy.samplers.KSampler.SAMPLERS
_VALID_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS

@VariantSupport()
class PrepareLatentDenoise(LatentNode):
    """Apply the correct ε·σ₀ noise to a latent and output the sigma ladder.

    Feed *latent_out* → SamplerCustom.latent_image
         *sigmas*     → SamplerCustom.sigmas
         set SamplerCustom.add_noise = False
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sampler_name": (_VALID_SAMPLERS,),
                "scheduler": (_VALID_SCHEDULERS,),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10_000}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "latent_in": ("LATENT",),
            },
            "optional": {
                "noise": ("LATENT",),          # ε batch; if None, random per‑frame noise
            },
        }

    RETURN_TYPES = ("LATENT", "SIGMAS", "INT")
    RETURN_NAMES = ("latent_out", "sigmas", "start_step")
    FUNCTION = "process"

    # ------------------------------------------------------------ core logic --
    def process(self, model, sampler_name, scheduler, steps, denoise,
                latent_in, noise=None):

        samples = latent_in["samples"].to("cpu")          # work on CPU

        # ── 0. short-circuit: denoise == 0  ────────────────────────────────
        if denoise <= 0.0:
            empty = torch.FloatTensor([])
            return (latent_in, empty, steps)              # unchanged latent

        # ── 1. schedule indices exactly like ComfyUI  ─────────────────────
        if denoise >= 0.9999:
            new_steps, start_at = steps, 0
        else:
            new_steps = int(steps / denoise)
            start_at  = new_steps - steps

        # ── 2. build sigma ladder  ────────────────────────────────────────
        device = comfy.model_management.get_torch_device()
        comfy.model_management.load_model_gpu(model)
        ks = comfy.samplers.KSampler(
            model, steps=new_steps, device=device,
            sampler=sampler_name, scheduler=scheduler,
            denoise=1.0, model_options=model.model_options,
        )
        sig_full  = ks.sigmas.to("cpu")
        sig_slice = sig_full[start_at : start_at + steps + 1].clone()
        sigma0_scaled = sig_slice[0] / model.model.latent_format.scale_factor

        # ── 3. optionally inject ε · σ₀  ──────────────────────────────────
        if noise is None:
            # Caller didn’t supply ε → keep latent unchanged
            latent_out = latent_in                      # no extra noise
        else:
            eps = noise["samples"]
            if eps.shape != samples.shape:
                raise ValueError("Noise batch shape does not match latent batch.")
            noised = samples + eps * sigma0_scaled
            latent_out = {"samples": noised.to("cpu")}

        return (latent_out, sig_slice, start_at)

@VariantSupport()
class LatentNoiseInjection(LatentNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "noise_std": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "inject_noise"

    def inject_noise(self, samples, noise_std):
        result = samples.copy()
        result["samples"] = result["samples"] + torch.randn_like(result["samples"]) * noise_std
        return (result,)

# -----------------------------------------------------------------------------
#  Registration tables
# -----------------------------------------------------------------------------
LATENT_NODE_CLASS_MAPPINGS = {
    f"SeedInterpNoise{NODE_POSTFIX}": SeedInterpNoise,
    f"PrepareLatentDenoise{NODE_POSTFIX}": PrepareLatentDenoise,
    f"LatentNoiseInjection{NODE_POSTFIX}": LatentNoiseInjection,
}

LATENT_NODE_DISPLAY_NAME_MAPPINGS = {
    f"SeedInterpNoise{NODE_POSTFIX}": f"Seed Interp Noise {NODE_POSTFIX}",
    f"PrepareLatentDenoise{NODE_POSTFIX}": f"Prepare Latent Denoise {NODE_POSTFIX}",
    f"LatentNoiseInjection{NODE_POSTFIX}": f"Latent Noise Injection {NODE_POSTFIX}",
}
