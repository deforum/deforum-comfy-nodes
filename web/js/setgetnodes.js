const { app } = window.comfyAPI.app;

// Deforum-owned frontend virtual Set/Get nodes. These nodes are intentionally
// JS-only: ComfyUI resolves them into real links before the prompt reaches the
// backend.
const SET_NODE_TYPE = "DeforumSetNode";
const GET_NODE_TYPE = "DeforumGetNode";
const AUX_ID = "deforum/deforum-comfy-nodes";
const LGraphNode = LiteGraph.LGraphNode;

const TYPE_COLORS = {
	DEFAULT: LGraphCanvas.node_colors?.gray,
	MODEL: LGraphCanvas.node_colors?.blue,
	LATENT: LGraphCanvas.node_colors?.purple,
	VAE: LGraphCanvas.node_colors?.red,
	CONDITIONING: LGraphCanvas.node_colors?.brown,
	IMAGE: LGraphCanvas.node_colors?.pale_blue,
	CLIP: LGraphCanvas.node_colors?.yellow,
	FLOAT: LGraphCanvas.node_colors?.green,
	INT: { color: "#1b4669", bgcolor: "#29699c" },
	BOOLEAN: { color: "#1b4669", bgcolor: "#29699c" },
	MASK: { color: "#1c5715", bgcolor: "#1f401b" },
	CONTROL_NET: { color: "#156653", bgcolor: "#1c453b" },
	NOISE: { color: "#2e2e2e", bgcolor: "#242121" },
	GUIDER: { color: "#3c7878", bgcolor: "#1c453b" },
	SAMPLER: { color: "#614a4a", bgcolor: "#3b2c2c" },
	SIGMAS: { color: "#485248", bgcolor: "#272e27" },
};

function getLink(graph, linkId) {
	if (linkId == null || !graph) return null;
	if (graph.links?.[linkId]) return graph.links[linkId];
	if (graph._links?.[linkId]) return graph._links[linkId];
	if (Array.isArray(graph.links)) return graph.links.find((link) => link?.id === linkId) ?? null;
	return null;
}

function getRootGraph(graph) {
	return graph?.rootGraph || app.graph || graph;
}

function collectGraphs(rootGraph) {
	const seen = new Set();
	const graphs = [];
	const visit = (graph) => {
		if (!graph || seen.has(graph)) return;
		seen.add(graph);
		graphs.push(graph);
		for (const node of graph._nodes || []) {
			if (node.subgraph) visit(node.subgraph);
		}
		const subgraphs = graph._subgraphs || graph.subgraphs;
		if (subgraphs?.values) {
			for (const subgraph of subgraphs.values()) visit(subgraph);
		}
	};
	visit(rootGraph);
	return graphs;
}

function findParentGraph(rootGraph, childGraph) {
	for (const graph of collectGraphs(rootGraph)) {
		for (const node of graph._nodes || []) {
			if (node.subgraph === childGraph) return graph;
		}
	}
	return null;
}

function getGraphScope(graph) {
	if (!graph) return [];
	const rootGraph = getRootGraph(graph);
	const scope = [graph];
	let current = graph;
	const seen = new Set(scope);
	while (current && current !== rootGraph) {
		const parent = findParentGraph(rootGraph, current);
		if (!parent || seen.has(parent)) break;
		scope.push(parent);
		seen.add(parent);
		current = parent;
	}
	return scope;
}

function nodeName(node) {
	return node?.widgets?.[0]?.value ?? "";
}

function nodesOfType(graph, type) {
	return (graph?._nodes || []).filter((node) => node.type === type);
}

function findSetterByName(graph, name) {
	if (!name) return null;
	for (const scopedGraph of getGraphScope(graph)) {
		const setter = nodesOfType(scopedGraph, SET_NODE_TYPE).find((node) => nodeName(node) === name);
		if (setter) return { node: setter, graph: scopedGraph };
	}
	return null;
}

function visibleSetterNames(graph) {
	const names = [];
	const seen = new Set();
	for (const scopedGraph of getGraphScope(graph)) {
		for (const node of nodesOfType(scopedGraph, SET_NODE_TYPE)) {
			const name = nodeName(node);
			if (name && !seen.has(name)) {
				seen.add(name);
				names.push(name);
			}
		}
	}
	return names.sort((a, b) => a.localeCompare(b));
}

function typeFromLink(graph, linkId) {
	const link = getLink(graph, linkId);
	if (!link) return "*";
	if (link.type && link.type !== "*") return link.type;
	const sourceNode = graph?.getNodeById?.(link.origin_id);
	const sourceOutput = sourceNode?.outputs?.[link.origin_slot];
	return sourceOutput?.type || "*";
}

function setterInputType(setter) {
	const input = setter?.inputs?.[0];
	if (!input) return "*";
	if (input.link != null) return typeFromLink(setter.graph, input.link);
	return input.type || "*";
}

function applyType(node, type, direction) {
	const cleanType = type || "*";
	const slot = direction === "input" ? node.inputs?.[0] : node.outputs?.[0];
	if (slot) {
		slot.name = cleanType;
		slot.type = cleanType;
	}
	const colors = TYPE_COLORS[cleanType] || TYPE_COLORS.DEFAULT;
	if (colors) {
		node.color = colors.color;
		node.bgcolor = colors.bgcolor;
	}
}

function isTypeCompatible(sourceType, targetType) {
	if (!sourceType || !targetType || sourceType === "*" || targetType === "*") return true;
	const targets = String(targetType).split(",");
	return targets.includes(sourceType) || String(sourceType).split(",").some((type) => targets.includes(type));
}

function markCanvasDirty() {
	app.canvas?.setDirty(true, true);
}

function updateGetter(getter) {
	const name = nodeName(getter);
	const result = findSetterByName(getter.graph, name);
	const type = result ? setterInputType(result.node) : "*";
	applyType(getter, type, "output");
	getter.title = name ? `Get_${name}` : "Get";

	const output = getter.outputs?.[0];
	if (getter.graph && output?.links?.length && type !== "*") {
		for (const linkId of [...output.links]) {
			const link = getLink(getter.graph, linkId);
			const target = getter.graph.getNodeById?.(link?.target_id);
			const targetType = target?.inputs?.[link?.target_slot]?.type;
			if (!isTypeCompatible(type, targetType)) getter.graph.removeLink(linkId);
		}
	}
}

function updateGettersForName(rootGraph, name) {
	if (!rootGraph || !name) return;
	for (const graph of collectGraphs(getRootGraph(rootGraph))) {
		for (const getter of nodesOfType(graph, GET_NODE_TYPE)) {
			if (nodeName(getter) === name) updateGetter(getter);
		}
	}
}

function ensureCommonProperties(node, nodeType) {
	node.properties = node.properties || {};
	node.properties["Node name for S&R"] = nodeType;
	node.properties.aux_id = AUX_ID;
	node.isVirtualNode = true;
	node.serialize_widgets = true;
	node.drawConnection = false;
}

app.registerExtension({
	name: "Deforum.SetGet.Nodes",
	registerCustomNodes() {
		class DeforumSetNode extends LGraphNode {
			static title = "Set";
			static category = "Deforum";

			constructor(title) {
				super(title);
				ensureCommonProperties(this, SET_NODE_TYPE);
				this.addWidget("text", "name", "", () => this.onRename());
				this.addInput("*", "*");
				this.addOutput("*", "*");
			}

			onConfigure() {
				ensureCommonProperties(this, SET_NODE_TYPE);
				setTimeout(() => this.updateType(), 0);
			}

			onAdded() {
				this.updateType();
			}

			onConnectionsChange() {
				if (!app.configuringGraph) this.updateType();
			}

			onRename() {
				this.updateType();
			}

			updateType() {
				ensureCommonProperties(this, SET_NODE_TYPE);
				const name = nodeName(this);
				const previousName = this.properties.previousName;
				this.properties.previousName = name;
				this.title = name ? `Set_${name}` : "Set";
				const type = setterInputType(this);
				applyType(this, type, "input");
				applyType(this, type, "output");
				updateGettersForName(this.graph, name);
				if (previousName && previousName !== name) updateGettersForName(this.graph, previousName);
				markCanvasDirty();
			}
		}

		class DeforumGetNode extends LGraphNode {
			static title = "Get";
			static category = "Deforum";

			constructor(title) {
				super(title);
				ensureCommonProperties(this, GET_NODE_TYPE);
				const options = {};
				Object.defineProperty(options, "values", {
					get: () => visibleSetterNames(this.graph),
					enumerable: true,
				});
				this.addWidget("combo", "name", "", () => this.onRename(), options);
				this.addOutput("*", "*");
			}

			onConfigure() {
				ensureCommonProperties(this, GET_NODE_TYPE);
				setTimeout(() => this.onRename(), 0);
			}

			onAdded() {
				this.onRename();
			}

			onConnectionsChange() {
				if (!app.configuringGraph) this.onRename();
			}

			onRename() {
				updateGetter(this);
				markCanvasDirty();
			}

			setName(name) {
				if (this.widgets?.[0]) this.widgets[0].value = name;
				this.onRename();
			}

			getInputLink(slot) {
				if (slot !== 0) return null;
				const name = nodeName(this);
				if (!name) return null;
				const setter = nodesOfType(this.graph, SET_NODE_TYPE).find((node) => nodeName(node) === name);
				const linkId = setter?.inputs?.[0]?.link;
				return getLink(this.graph, linkId);
			}

			resolveVirtualOutput(slot) {
				if (slot !== 0) return undefined;
				const name = nodeName(this);
				const result = findSetterByName(this.graph, name);
				if (!result || result.graph === this.graph) return undefined;

				const link = getLink(result.graph, result.node.inputs?.[0]?.link);
				if (!link) return undefined;
				const sourceNode = result.graph.getNodeById?.(link.origin_id);
				if (!sourceNode) return undefined;
				return { node: sourceNode, slot: link.origin_slot };
			}

			getExtraMenuOptions(_, options) {
				if (!options) return;
				options.push({
					content: "Jump to setter",
					callback: () => {
						const result = findSetterByName(this.graph, nodeName(this));
						if (!result?.node) return;
						if (result.graph && result.graph !== this.graph) app.canvas?.setGraph?.(result.graph);
						app.canvas?.centerOnNode?.(result.node);
						app.canvas?.selectNode?.(result.node, false);
					},
				});
			}
		}

		LiteGraph.registerNodeType(SET_NODE_TYPE, DeforumSetNode);
		LiteGraph.registerNodeType(GET_NODE_TYPE, DeforumGetNode);
	},
});
