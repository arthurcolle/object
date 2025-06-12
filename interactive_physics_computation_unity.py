"""
Physics-Computation Unity: A Clear Conceptual Framework
======================================================

This visualization demonstrates the mathematical unity between physics,
computation, and consciousness through clear, intuitive representations.
"""

import math
import os
import webbrowser
import plotly.graph_objects as go
import plotly.express as px

def create_conceptual_framework():
    """Create a clear, meaningful visualization of physics-computation unity."""
    
    print("🔬 Creating conceptual framework visualization...")
    
    fig = go.Figure()
    
    # Central concept: The Unity Sphere
    print("🌐 Building central unity concept...")
    
    # Create a sphere representing the unified framework
    phi = [i * 0.2 for i in range(32)]  # Latitude
    theta = [j * 0.4 for j in range(16)]  # Longitude
    
    x_sphere, y_sphere, z_sphere = [], [], []
    for p in phi:
        for t in theta:
            x = 3 * math.sin(p) * math.cos(t)
            y = 3 * math.sin(p) * math.sin(t) 
            z = 3 * math.cos(p)
            x_sphere.append(x)
            y_sphere.append(y)
            z_sphere.append(z)
    
    fig.add_trace(go.Scatter3d(
        x=x_sphere, y=y_sphere, z=z_sphere,
        mode='markers',
        marker=dict(
            size=2,
            opacity=0.3,
            color='rgba(100,100,200,0.5)'
        ),
        name='Unity Framework',
        hoverinfo='skip'
    ))
    
    # The Three Pillars
    print("🏛️ Creating the three pillars...")
    
    # 1. PHYSICS PILLAR (Left)
    physics_concepts = [
        ("Quantum Fields", -8, 0, 8),
        ("Spacetime", -8, 0, 4), 
        ("Energy-Matter", -8, 0, 0),
        ("Thermodynamics", -8, 0, -4),
        ("Information", -8, 0, -8)
    ]
    
    for name, x, y, z in physics_concepts:
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=15, color='red'),
            text=[name],
            textposition='middle right',
            name='Physics',
            legendgroup='physics',
            showlegend=(name == physics_concepts[0][0])
        ))
    
    # Connect physics concepts
    for i in range(len(physics_concepts)-1):
        fig.add_trace(go.Scatter3d(
            x=[physics_concepts[i][1], physics_concepts[i+1][1], None],
            y=[physics_concepts[i][2], physics_concepts[i+1][2], None],
            z=[physics_concepts[i][3], physics_concepts[i+1][3], None],
            mode='lines',
            line=dict(width=4, color='red'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 2. COMPUTATION PILLAR (Right)
    computation_concepts = [
        ("Lambda Calculus", 8, 0, 8),
        ("Type Theory", 8, 0, 4),
        ("Category Theory", 8, 0, 0),
        ("Logic Systems", 8, 0, -4),
        ("Algorithms", 8, 0, -8)
    ]
    
    for name, x, y, z in computation_concepts:
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=15, color='blue'),
            text=[name],
            textposition='middle left',
            name='Computation',
            legendgroup='computation',
            showlegend=(name == computation_concepts[0][0])
        ))
    
    # Connect computation concepts
    for i in range(len(computation_concepts)-1):
        fig.add_trace(go.Scatter3d(
            x=[computation_concepts[i][1], computation_concepts[i+1][1], None],
            y=[computation_concepts[i][2], computation_concepts[i+1][2], None],
            z=[computation_concepts[i][3], computation_concepts[i+1][3], None],
            mode='lines',
            line=dict(width=4, color='blue'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 3. CONSCIOUSNESS PILLAR (Front)
    consciousness_concepts = [
        ("Integrated Information", 0, 8, 8),
        ("Emergence", 0, 8, 4),
        ("Self-Organization", 0, 8, 0),
        ("Complexity", 0, 8, -4),
        ("Awareness", 0, 8, -8)
    ]
    
    for name, x, y, z in consciousness_concepts:
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=15, color='green'),
            text=[name],
            textposition='middle center',
            name='Consciousness',
            legendgroup='consciousness',
            showlegend=(name == consciousness_concepts[0][0])
        ))
    
    # Connect consciousness concepts
    for i in range(len(consciousness_concepts)-1):
        fig.add_trace(go.Scatter3d(
            x=[consciousness_concepts[i][1], consciousness_concepts[i+1][1], None],
            y=[consciousness_concepts[i][2], consciousness_concepts[i+1][2], None],
            z=[consciousness_concepts[i][3], consciousness_concepts[i+1][3], None],
            mode='lines',
            line=dict(width=4, color='green'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # BRIDGE CONNECTIONS - showing the unity
    print("🌉 Creating conceptual bridges...")
    
    bridges = [
        # Physics ↔ Computation
        (physics_concepts[0], computation_concepts[0], 'purple'),  # Quantum ↔ Lambda
        (physics_concepts[1], computation_concepts[1], 'purple'),  # Spacetime ↔ Types
        (physics_concepts[2], computation_concepts[2], 'purple'),  # Energy ↔ Category
        
        # Physics ↔ Consciousness  
        (physics_concepts[2], consciousness_concepts[0], 'orange'), # Energy ↔ Information
        (physics_concepts[3], consciousness_concepts[1], 'orange'), # Thermo ↔ Emergence
        
        # Computation ↔ Consciousness
        (computation_concepts[0], consciousness_concepts[0], 'cyan'), # Lambda ↔ Information
        (computation_concepts[1], consciousness_concepts[1], 'cyan'), # Types ↔ Emergence
    ]
    
    for (concept1, concept2, color) in bridges:
        fig.add_trace(go.Scatter3d(
            x=[concept1[1], concept2[1], None],
            y=[concept1[2], concept2[2], None], 
            z=[concept1[3], concept2[3], None],
            mode='lines',
            line=dict(width=3, color=color, dash='dot'),
            showlegend=False,
            hoverinfo='skip',
            opacity=0.7
        ))
    
    # Add central equations/principles
    print("📐 Adding fundamental equations...")
    
    equations = [
        ("E = mc²", 0, 0, 6, 'Physics'),
        ("λx.M", 0, 0, 2, 'Computation'),  
        ("Φ = ∫ φ", 0, 0, -2, 'Information'),
        ("S = k ln W", 0, 0, -6, 'Entropy')
    ]
    
    for eq, x, y, z, domain in equations:
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=12, color='gold', symbol='diamond'),
            text=[eq],
            textposition='middle center',
            name='Core Equations',
            legendgroup='equations',
            showlegend=(eq == equations[0][0])
        ))
    
    # LAYOUT
    fig.update_layout(
        title={
            'text': "Physics-Computation Unity: A Clear Conceptual Framework<br><sub>Three Pillars of Reality Connected by Fundamental Principles</sub>",
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis=dict(
                title="Physics ← → Computation",
                showgrid=True,
                gridcolor='rgba(200,200,200,0.3)',
                range=[-12, 12]
            ),
            yaxis=dict(
                title="← Consciousness",
                showgrid=True, 
                gridcolor='rgba(200,200,200,0.3)',
                range=[-2, 12]
            ),
            zaxis=dict(
                title="Abstraction Level",
                showgrid=True,
                gridcolor='rgba(200,200,200,0.3)',
                range=[-10, 10]
            ),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            bgcolor='rgba(255,255,255,0.9)'
        ),
        height=800,
        width=1200,
        font=dict(size=10),
        legend=dict(
            x=0.02, 
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )
    
    return fig

def main(open_browser=True):
    """Generate the clear conceptual visualization."""
    
    fig = create_conceptual_framework()
    
    # Save to HTML
    html_file = "physics_computation_unity.html"
    fig.write_html(html_file, auto_open=False)
    
    file_path = os.path.abspath(html_file)
    print(f"✅ Visualization saved to: {file_path}")
    print(f"")
    print(f"🎯 CONCEPTUAL FRAMEWORK:")
    print(f"   📍 Three Pillars: Physics (red), Computation (blue), Consciousness (green)")
    print(f"   🌉 Bridge Connections: Purple, Orange, Cyan dotted lines")
    print(f"   💎 Core Equations: Gold diamonds at center")
    print(f"   🌐 Unity Sphere: Semi-transparent background structure")
    print(f"")
    print(f"🎮 Controls: Drag to rotate, scroll to zoom, hover for details")
    print(f"📐 Layout: Clear spatial organization with meaningful axes")
    
    if open_browser:
        webbrowser.open_new_tab(file_path)
        print("🌐 Opening in browser...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Physics-Computation Unity Conceptual Framework")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()
    
    main(open_browser=not args.no_browser)