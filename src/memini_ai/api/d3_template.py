"""D3.js HTML template for live knowledge graph visualization.

Generates a self-contained HTML page that polls the FastAPI server
every 30 seconds to fetch updated graph data from PostgreSQL.
"""

from __future__ import annotations


def generate_live_html(poll_interval_ms: int = 30000) -> str:
    """Generate self-contained HTML page with live-updating D3.js graph.

    Args:
        poll_interval_ms: How often to refresh the graph (default 30 seconds).

    Returns:
        Complete HTML string with embedded JavaScript.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memini-ai Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --border-color: #2d3a5a;
            --accent: #4a90d9;
        }}
        @media (prefers-color-scheme: light) {{
            :root {{
                --bg-primary: #f5f5f5;
                --bg-secondary: #ffffff;
                --text-primary: #1a1a2e;
                --text-secondary: #4a4a6a;
                --border-color: #d0d0e0;
            }}
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{ font-size: 1.25rem; font-weight: 600; }}
        .stats {{
            display: flex;
            gap: 1.5rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }}
        .stat-item {{ display: flex; align-items: center; gap: 0.5rem; }}
        .stat-value {{ color: var(--accent); font-weight: 600; }}
        .controls {{
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }}
        button {{
            background: var(--accent);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.875rem;
            transition: opacity 0.2s;
        }}
        button:hover {{ opacity: 0.85; }}
        .status {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        .status.live {{ color: #27ae60; }}
        .status.error {{ color: #e74c3c; }}
        #graph-container {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1rem;
            position: relative;
        }}
        svg {{ background: var(--bg-secondary); border-radius: 8px; }}
        .node {{ cursor: pointer; stroke-width: 2px; }}
        .node:hover {{ stroke: #fff; stroke-width: 3px; }}
        .link {{ stroke-opacity: 0.6; }}
        .link:hover {{ stroke-opacity: 1; }}
        .tooltip {{
            position: absolute;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.75rem;
            font-size: 0.8rem;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 280px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 100;
        }}
        .tooltip.visible {{ opacity: 1; }}
        .tooltip h3 {{ font-size: 0.9rem; margin-bottom: 0.5rem; }}
        .tooltip p {{ margin: 0.25rem 0; color: var(--text-secondary); }}
        .tooltip .type {{
            display: inline-block;
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 500;
            color: white;
        }}
        .legend {{
            position: absolute;
            bottom: 1rem;
            right: 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1rem;
            font-size: 0.75rem;
        }}
        .legend h4 {{ margin-bottom: 0.5rem; font-weight: 600; }}
        .legend-item {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.25rem 0; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 50%; }}
        .no-data {{
            text-align: center;
            padding: 4rem;
            color: var(--text-secondary);
        }}
        .loading {{ text-align: center; padding: 2rem; color: var(--text-secondary); }}
    </style>
</head>
<body>
    <header>
        <h1>🧠 Knowledge Graph Visualization</h1>
        <div class="stats">
            <div class="stat-item">
                <span>Nodes:</span>
                <span class="stat-value" id="node-count">0</span>
            </div>
            <div class="stat-item">
                <span>Edges:</span>
                <span class="stat-value" id="edge-count">0</span>
            </div>
            <div class="stat-item">
                <span class="status live" id="status">● LIVE</span>
            </div>
        </div>
        <div class="controls">
            <button onclick="refreshGraph()">🔄 Refresh</button>
        </div>
    </header>
    <div id="graph-container">
        <div class="loading" id="loading">Loading graph data...</div>
    </div>
    <div class="tooltip" id="tooltip">
        <h3 id="tooltip-name"></h3>
        <p><span class="type" id="tooltip-type"></span></p>
        <p>Confidence: <span id="tooltip-confidence"></span></p>
        <p>Mentions: <span id="tooltip-mentions"></span></p>
        <p id="tooltip-rels"></p>
    </div>
    <div class="legend">
        <h4>Node Types</h4>
        <div class="legend-item"><div class="legend-color" style="background:#4a90d9"></div>Person</div>
        <div class="legend-item"><div class="legend-color" style="background:#27ae60"></div>Organization</div>
        <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div>Concept</div>
        <div class="legend-item"><div class="legend-color" style="background:#e67e22"></div>Code</div>
        <div class="legend-item"><div class="legend-color" style="background:#f1c40f"></div>Project</div>
        <div class="legend-item"><div class="legend-color" style="background:#1abc9c"></div>Location</div>
        <div class="legend-item"><div class="legend-color" style="background:#95a5a6"></div>Unknown</div>
        <h4 style="margin-top:0.75rem">Relationships</h4>
        <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div>Supersedes</div>
        <div class="legend-item"><div class="legend-color" style="background:#3498db"></div>Related To</div>
        <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div>Contradicts</div>
        <div class="legend-item"><div class="legend-color" style="background:#27ae60"></div>Derived From</div>
    </div>

    <script>
    const API_BASE = '/api';
    const POLL_INTERVAL = {poll_interval_ms};
    let nodes = [];
    let edges = [];
    let simulation = null;

    const typeColors = {{
        PERSON: "#4a90d9",
        ORGANIZATION: "#27ae60",
        CONCEPT: "#9b59b6",
        CODE: "#e67e22",
        PROJECT: "#f1c40f",
        LOCATION: "#1abc9c",
        UNKNOWN: "#95a5a6"
    }};

    async function fetchGraph() {{
        try {{
            const response = await fetch("/api/graph?limit=500");
            if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
            const data = await response.json();
            nodes = data.nodes || [];
            edges = data.edges || [];
            updateStats();
            renderGraph();
            document.getElementById('status').textContent = '● LIVE';
            document.getElementById('status').className = 'status live';
        }} catch (error) {{
            console.error('Failed to fetch graph:', error);
            document.getElementById('status').textContent = '● ERROR';
            document.getElementById('status').className = 'status error';
        }}
    }}

    function updateStats() {{
        document.getElementById('node-count').textContent = nodes.length;
        document.getElementById('edge-count').textContent = edges.length;
    }}

    function renderGraph() {{
        const container = document.getElementById('graph-container');

        // Clear previous content
        container.innerHTML = '';

        if (nodes.length === 0) {{
            container.innerHTML = '<div class="no-data">No graph data available. Add entities to your knowledge graph to see them visualized here.</div>';
            return;
        }}

        const width = Math.min(container.clientWidth - 32, 1400);
        const height = Math.min(window.innerHeight - 140, 900);

        const svg = d3.select("#graph-container")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        const g = svg.append("g");

        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => g.attr("transform", event.transform));
        svg.call(zoom);

        // Arrow marker
        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-5 -5 10 10")
            .attr("refX", 25)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M-5,-5L5,0L-5,5")
            .attr("fill", "#666");

        // Create simulation
        simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(edges).id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(50));

        // Draw links
        const link = g.append("g")
            .selectAll("line")
            .data(edges)
            .join("line")
            .attr("class", "link")
            .attr("stroke", d => d.stroke || "#666")
            .attr("stroke-width", d => Math.max(1, d.confidence * 3))
            .attr("marker-end", "url(#arrowhead)");

        // Draw nodes
        const node = g.append("g")
            .selectAll("g")
            .data(nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", (event, d) => {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                }})
                .on("drag", (event, d) => {{ d.fx = event.x; d.fy = event.y; }})
                .on("end", (event, d) => {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null; d.fy = null;
                }}));

        node.append("circle")
            .attr("r", d => Math.max(10, Math.min(30, 10 + (d.mention_count || d.mentions || 0) * 2)))
            .attr("fill", d => typeColors[d.type] || typeColors.UNKNOWN)
            .attr("stroke", "#fff");

        node.append("text")
            .text(d => d.name.length > 18 ? d.name.substring(0, 15) + "..." : d.name)
            .attr("text-anchor", "middle")
            .attr("dy", d => Math.max(12, Math.min(35, 12 + (d.mention_count || d.mentions || 0) * 2)) + 18)
            .attr("fill", "var(--text-primary)")
            .attr("font-size", "11px");

        // Tooltip
        const tooltip = document.getElementById("tooltip");
        const tooltipName = document.getElementById("tooltip-name");
        const tooltipType = document.getElementById("tooltip-type");
        const tooltipConfidence = document.getElementById("tooltip-confidence");
        const tooltipMentions = document.getElementById("tooltip-mentions");
        const tooltipRels = document.getElementById("tooltip-rels");

        node.on("mouseover", (event, d) => {{
            tooltipName.textContent = d.name;
            tooltipType.textContent = d.type;
            tooltipType.style.background = typeColors[d.type] || typeColors.UNKNOWN;
            tooltipConfidence.textContent = ((d.confidence || 1) * 100).toFixed(0) + "%";
            tooltipMentions.textContent = d.mention_count || d.mentions || 0;

            const relatedEdges = edges.filter(e => e.source.id === d.id || e.target.id === d.id);
            tooltipRels.textContent = (relatedEdges.length || 0) + " relationships";
            tooltip.classList.add("visible");
        }})
        .on("mousemove", (event) => {{
            tooltip.style.left = (event.pageX + 15) + "px";
            tooltip.style.top = (event.pageY - 10) + "px";
        }})
        .on("mouseout", () => tooltip.classList.remove("visible"));

        // Simulation tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});
    }}

    function refreshGraph() {{
        const status = document.getElementById('status');
        status.textContent = '● REFRESHING...';
        status.className = 'status';
        fetchGraph();
    }}

    // Initial load
    fetchGraph();

    // Auto-refresh
    setInterval(fetchGraph, POLL_INTERVAL);
    </script>
</body>
</html>
"""
