# Deforum Flux Complex Workflow

![wf_overview](https://github.com/user-attachments/assets/725d4351-2c39-42e6-8ac3-f9dd8216b1ac)


## Credits:

[Huemin:](https://huemin.art/) Creator of [Deforum](https://deforum.art/) and [Dream Computing](https://www.dreamcomputing.io/)

[Akatz:](https://akatz.ai/) Workflow builder and founder of [ComfyGit](https://github.com/comfygit-ai)


## Step 0. Download Nodes

You'll need these custom nodes for the workflow to work properly:

Download via the Manager (search for IDs below) or via Github:

[comfyui-basic-math](https://github.com/akatz-ai/ComfyUI-Basic-Math)

[comfyui-deforum](https://github.com/deforum/deforum-comfy-nodes)


## Step 1. Download models

Required:

Save under checkpoints/
[flux1-dev-fp8.safetensors](https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors?download=true)

Optional (but recommended):

Save under loras/
[Hyper-FLUX.1-dev-8steps-lora.safetensors](https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors?download=true)

Save under loras/
[FLUX-dev-lora-AntiBlur.safetensors](https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur/resolve/main/FLUX-dev-lora-AntiBlur.safetensors?download=true)

Save under loras/
[Flux.1-Turbo-Alpha](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/tree/main)


## Step 2. Configure Inputs

![inputs](https://github.com/user-attachments/assets/87e51941-747c-4803-9bb1-fd03b99c7c52)

**Seed**: The initial seed used in generating

**Num Frames**: How many total frames the final animation should have

**Width**: The width of the images

**Height**: The height of the images

**Default Steps:** The default number of sampling steps that will be used if a step schedule is not present and enabled.

**Default Denoise:** The default denoise setting that will be used if a denoise schedule is not present and enabled.

**Default CFG:** The default Classifier Free Guidance setting that will be used if a CFG schedule is not present and enabled.

**Lora Loader Stack:** Where you can load FLUX loras to influence the final result.

**Use Upload Image?:** A switch that if set to TRUE will use the uploaded image as the initial frame in the generation sequence, and if FALSE will generate a random starting frame based on the prompt and seed.


## Step 3. Configure Prompt Schedules

![prompt_schedules](https://github.com/user-attachments/assets/d7c570fb-685d-4a52-9512-92fba22376c6)

The "Schedules" group has a Prompt Schedule node which will let you configure the prompt (or sequence of prompts) using frame indices for the duration of the animation sequence.

These Schedule nodes follow the string format as FizzNodes: [documentation from the same github repo](https://github.com/FizzleDorf/ComfyUI_FizzNodes/wiki/Prompt-Schedules)


## Step 4. Configure 2D Motion

![2d_motion](https://github.com/user-attachments/assets/0ba788b6-462b-41e4-9a9a-e08d5f5e9b8b)

You can modify the values in the **Transform Image** group above to create dramatic effects in the animation, such as infinite zoom or horizontal and vertical traveling.

Edit the **Scale** float value to influence zoom speed.

Edit the **X and Y** float values to influence horizontal and vertical movement speed respectively.


## Step 5. (Optional) Enable and Configure Value Schedules

![value_schedules](https://github.com/user-attachments/assets/8f73bbc7-253c-487b-94c8-d76bad0a00a8)

The bypassed groups below the Input and Schedules groups can be used to vary the Denoise, Step count, Seed value, and CFG values over the duration of the animation.

Each group has a "Fast Bypasser" node which you can click on to un-bypass the group, or you can use the "Fast Groups Bypasser" node above to enable/disable them.

Each Schedule group has a Value Schedule node which can be used to change values over time based on a frame index and number.


## Step 6. RUN the workflow

Your final generation result will be visible in the Outputs group below:

You can change the frame_rate and filename_prefix as desired.

![outputs](https://github.com/user-attachments/assets/a0697562-65ee-4cc6-8ede-dae532a213d2)

## Step 7. Profit

## Extras:

### Sampling:

![sampling](https://github.com/user-attachments/assets/1e576d0d-9941-49b2-903d-4788af44727d)

This is the group where the FLUX sampler lives. Inputs to the group can be seen on the left as an array of GET nodes, and outputs can be seen on the right as SET nodes.

Generally the “final” values are fed into the sampler, and are set in the “Utils” group seen below.

### Generated Seed List:

![generated_seed_list](https://github.com/user-attachments/assets/7f7f2fbb-24ae-4506-a084-16c082b164ab)

This group is where this list of seeds for generating noise each frame will be generated.

You can change the “control_mode” to **random** instead of **increment** to get a list of random seed values (including your initial seed set vai the Inputs group).

### Open Loop:

![open_loop](https://github.com/user-attachments/assets/32dfdcdb-7f88-4244-8724-ce743ee3bc6a)

This is the group containing the opening For Loop node, and is responsible for providing the current loop index, a boolean if the current frame is the “start” index (first frame), and additional values passed via the “values” slots 1-4. 

These values exactly correspond to the For Loop Close group’s value slots 1-4. Any value (can be any type) passed to the For Loop Close value 1 (for example) will be present in the next iteration of the loop in For Loop Open value 1, and so on.

### Close Loop:

![close_loop](https://github.com/user-attachments/assets/8476824f-3a1e-4f24-8187-1a216e4d50d2)

Then end point of the For Loop for each iteration. Values passed to “initial_values” 1-4 will be present in the next iteration in For Loop Open’s values 1-4.
Once the loop is complete, any remaining values in the “initial_values” slots will pass to the output value slots, and execution will continue.

## Utility Groups:

### Fast Groups Bypasser:

![fast_groups_bypasser](https://github.com/user-attachments/assets/46e5c852-7500-42b6-a41e-e414c719e3bd)

Use the Fast Groups Bypasser node to selectively bypass groups remotely

### Initial Denoise

![init_denoise](https://github.com/user-attachments/assets/067860f0-5823-4672-80ea-eacc8525c070)

This utility group will set the initial denoise value to 1.0 (regardless of the default denoise or denoise schedule values) if an upload image is NOT used. This is to ensure the first frame is of high quality for subsequent generation frames.

### Util Switches

![util_switches](https://github.com/user-attachments/assets/b8252ca1-4146-4472-8495-d667c52843c3)

The set of switches in the Utils group above are used as a buffer between nodes and groups that transform values (like seed values, latent values, conditioning, etc.) and the final connection to the Sampling group. This is useful if you want to be able to bypass group nodes, and have “fallback” values in place (i.e. use a default seed value if we mute the seed scheduler group instead of crashing)

The any_01 slot is first priority in the switch, followed by any_02, any_03, etc.

For example, if the value in any_01 is found to be None or Null (bypassed), then control will pass to any_02 as a fallback. If any_02 is also invalid then control passes to any_03. And so on.
