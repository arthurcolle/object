#!/usr/bin/env python3
"""
Interactive AAOS (Autonomous Agency Operating System) Architecture Diagram
Comprehensive system visualization with all mechanisms and data flows
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math
import json

class AAOSSystemDiagram:
    def __init__(self, root):
        self.root = root
        self.root.title("AAOS - Autonomous Agency Operating System Architecture")
        self.root.geometry("1600x1000")
        
        # Colors for different component types
        self.colors = {
            'core': '#2E4057',
            'learning': '#048A81', 
            'communication': '#54C6EB',
            'security': '#F67E7D',
            'network': '#6C5B7B',
            'monitoring': '#FBB13C',
            'emergence': '#73A857',
            'storage': '#A8DADC',
            'agent': '#457B9D',
            'control': '#E63946'
        }
        
        self.setup_ui()
        self.components = self.initialize_components()
        self.connections = self.initialize_connections()
        self.draw_diagram()
        
    def setup_ui(self):
        # Main frame with canvas and scrollbars
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        self.canvas = tk.Canvas(self.main_frame, bg='#1a1a1a', width=1400, height=900)
        
        v_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.canvas.configure(scrollregion=(0, 0, 2000, 1500))
        
        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Control panel
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Layer controls
        ttk.Label(self.control_frame, text="System Layers", font=('Arial', 12, 'bold')).pack(pady=5)
        
        self.layer_vars = {}
        layers = ['Core', 'Learning', 'Communication', 'Security', 'Network', 'Monitoring', 'Emergence', 'Agents', 'Storage']
        
        for layer in layers:
            var = tk.BooleanVar(value=True)
            self.layer_vars[layer.lower()] = var
            cb = ttk.Checkbutton(self.control_frame, text=layer, variable=var, 
                               command=self.update_visibility)
            cb.pack(anchor=tk.W)
        
        # Info panel
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(self.control_frame, text="Component Info", font=('Arial', 12, 'bold')).pack(pady=5)
        
        self.info_text = tk.Text(self.control_frame, width=35, height=20, wrap=tk.WORD, font=('Courier', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        
        self.selected_component = None
        self.drag_data = {"x": 0, "y": 0}
        
    def initialize_components(self):
        """Define all AAOS system components with positions and metadata"""
        return {
            # Core System Layer
            'object_core': {
                'name': 'Object Core',
                'pos': (200, 200),
                'size': (120, 80),
                'type': 'core',
                'description': 'Base Object system with state, methods, goals, world models, and interaction history. Foundation for all autonomous agents.',
                'interfaces': ['state_management', 'method_dispatch', 'goal_planning'],
                'layer': 'core'
            },
            'meta_dsl': {
                'name': 'Meta-DSL',
                'pos': (350, 200),
                'size': (100, 60),
                'type': 'core',
                'description': 'Self-modification capabilities with :define, :goal, :belief, :infer constructs. Enables runtime adaptation.',
                'interfaces': ['self_modification', 'belief_system', 'inference_engine'],
                'layer': 'core'
            },
            'system_orchestrator': {
                'name': 'System Orchestrator',
                'pos': (500, 200),
                'size': (140, 80),
                'type': 'control',
                'description': 'LLM-powered self-organizing system management with dynamic topology discovery and fault tolerance.',
                'interfaces': ['topology_management', 'load_balancing', 'fault_recovery'],
                'layer': 'core'
            },
            
            # Agent Types Layer
            'ai_agent': {
                'name': 'AI Agent',
                'pos': (100, 350),
                'size': (90, 60),
                'type': 'agent',
                'description': 'Advanced reasoning agent with meta-learning and adaptation capabilities.',
                'interfaces': ['reasoning', 'meta_learning', 'adaptation'],
                'layer': 'agents'
            },
            'coordinator_object': {
                'name': 'Coordinator',
                'pos': (220, 350),
                'size': (90, 60),
                'type': 'agent',
                'description': 'Multi-object coordination and resource management agent.',
                'interfaces': ['coordination', 'resource_management', 'task_allocation'],
                'layer': 'agents'
            },
            'sensor_object': {
                'name': 'Sensor',
                'pos': (340, 350),
                'size': (80, 60),
                'type': 'agent',
                'description': 'Environmental monitoring with data preprocessing capabilities.',
                'interfaces': ['sensing', 'data_preprocessing', 'event_detection'],
                'layer': 'agents'
            },
            'actuator_object': {
                'name': 'Actuator',
                'pos': (450, 350),
                'size': (80, 60),
                'type': 'agent',
                'description': 'Physical control with motion planning and safety monitoring.',
                'interfaces': ['actuation', 'motion_planning', 'safety_monitoring'],
                'layer': 'agents'
            },
            'human_client': {
                'name': 'Human Client',
                'pos': (560, 350),
                'size': (90, 60),
                'type': 'agent',
                'description': 'Natural language interface with preference learning.',
                'interfaces': ['natural_language', 'preference_learning', 'human_interaction'],
                'layer': 'agents'
            },
            
            # Communication Layer
            'message_router': {
                'name': 'Message Router',
                'pos': (150, 500),
                'size': (120, 70),
                'type': 'communication',
                'description': 'GenStage-based routing with priority queuing, dead letter handling, and circuit breaker patterns.',
                'interfaces': ['priority_routing', 'backpressure', 'circuit_breaker'],
                'layer': 'communication'
            },
            'network_transport': {
                'name': 'Network Transport',
                'pos': (300, 500),
                'size': (130, 70),
                'type': 'communication',
                'description': 'Multi-protocol support (TCP/UDP/WebSocket/gRPC) with connection pooling and TLS encryption.',
                'interfaces': ['multi_protocol', 'connection_pool', 'encryption'],
                'layer': 'communication'
            },
            'mailbox_system': {
                'name': 'Mailbox System',
                'pos': (460, 500),
                'size': (110, 70),
                'type': 'communication',
                'description': 'Communication infrastructure with message routing and dyad formation for sustained cooperation.',
                'interfaces': ['message_routing', 'dyad_formation', 'interaction_history'],
                'layer': 'communication'
            },
            
            # Learning Layer
            'oorl_framework': {
                'name': 'OORL Framework',
                'pos': (100, 650),
                'size': (130, 80),
                'type': 'learning',
                'description': 'Object-Oriented RL with individual, social, collective, and meta-learning capabilities.',
                'interfaces': ['policy_learning', 'social_learning', 'meta_learning'],
                'layer': 'learning'
            },
            'collective_learning': {
                'name': 'Collective Learning',
                'pos': (260, 650),
                'size': (120, 80),
                'type': 'learning',
                'description': 'Coalition-based distributed optimization with swarm intelligence and collective decision-making.',
                'interfaces': ['coalition_formation', 'swarm_consensus', 'distributed_optimization'],
                'layer': 'learning'
            },
            'distributed_training': {
                'name': 'Distributed Training',
                'pos': (410, 650),
                'size': (130, 80),
                'type': 'learning',
                'description': 'Federated learning with Byzantine fault tolerance and adaptive optimization.',
                'interfaces': ['federated_learning', 'byzantine_tolerance', 'adaptive_optimization'],
                'layer': 'learning'
            },
            'transfer_learning': {
                'name': 'Transfer Learning',
                'pos': (570, 650),
                'size': (120, 80),
                'type': 'learning',
                'description': 'Cross-domain knowledge transfer and meta-learning for rapid adaptation.',
                'interfaces': ['knowledge_transfer', 'domain_adaptation', 'rapid_learning'],
                'layer': 'learning'
            },
            
            # Network Layer
            'p2p_bootstrap': {
                'name': 'P2P Bootstrap',
                'pos': (750, 300),
                'size': (110, 70),
                'type': 'network',
                'description': 'DHT-based peer discovery with gossip protocol and NAT traversal.',
                'interfaces': ['peer_discovery', 'gossip_protocol', 'nat_traversal'],
                'layer': 'network'
            },
            'distributed_registry': {
                'name': 'Distributed Registry',
                'pos': (750, 400),
                'size': (120, 70),
                'type': 'network',
                'description': 'Distributed consensus for object registration with heartbeat monitoring.',
                'interfaces': ['consensus', 'registration', 'heartbeat'],
                'layer': 'network'
            },
            'network_supervisor': {
                'name': 'Network Supervisor',
                'pos': (750, 500),
                'size': (120, 70),
                'type': 'network',
                'description': 'Network topology management and fault detection with automatic recovery.',
                'interfaces': ['topology_management', 'fault_detection', 'auto_recovery'],
                'layer': 'network'
            },
            
            # Security Layer
            'encryption': {
                'name': 'Encryption',
                'pos': (950, 250),
                'size': (100, 60),
                'type': 'security',
                'description': 'X25519 ECDH, Ed25519 signatures, ChaCha20-Poly1305 encryption, Double Ratchet, onion routing.',
                'interfaces': ['key_exchange', 'digital_signatures', 'authenticated_encryption'],
                'layer': 'security'
            },
            'byzantine_fault_tolerance': {
                'name': 'Byzantine FT',
                'pos': (950, 330),
                'size': (110, 70),
                'type': 'security',
                'description': 'PBFT-inspired consensus with reputation system and Merkle tree verification.',
                'interfaces': ['consensus', 'reputation_system', 'integrity_verification'],
                'layer': 'security'
            },
            'trust_manager': {
                'name': 'Trust Manager',
                'pos': (950, 420),
                'size': (100, 60),
                'type': 'security',
                'description': 'Reputation-based trust with proof-of-work and audit trails.',
                'interfaces': ['reputation_tracking', 'trust_computation', 'audit_trails'],
                'layer': 'security'
            },
            
            # Monitoring Layer
            'performance_monitor': {
                'name': 'Performance Monitor',
                'pos': (750, 650),
                'size': (130, 70),
                'type': 'monitoring',
                'description': 'Real-time metrics with adaptive thresholds and performance prediction.',
                'interfaces': ['metrics_collection', 'adaptive_thresholds', 'performance_prediction'],
                'layer': 'monitoring'
            },
            'resource_manager': {
                'name': 'Resource Manager',
                'pos': (950, 650),
                'size': (120, 70),
                'type': 'monitoring',
                'description': 'Dynamic resource allocation with load balancing and capacity planning.',
                'interfaces': ['resource_allocation', 'load_balancing', 'capacity_planning'],
                'layer': 'monitoring'
            },
            'agent_monitor': {
                'name': 'Agent Monitor',
                'pos': (1120, 650),
                'size': (100, 70),
                'type': 'monitoring',
                'description': 'Agent lifecycle and behavior monitoring with health checks.',
                'interfaces': ['lifecycle_monitoring', 'behavior_analysis', 'health_checks'],
                'layer': 'monitoring'
            },
            
            # Storage Layer
            'schema_registry': {
                'name': 'Schema Registry',
                'pos': (1200, 300),
                'size': (110, 60),
                'type': 'storage',
                'description': 'Centralized schema storage with versioning and compatibility checking.',
                'interfaces': ['schema_storage', 'versioning', 'compatibility_check'],
                'layer': 'storage'
            },
            'schema_evolution': {
                'name': 'Schema Evolution',
                'pos': (1200, 400),
                'size': (120, 70),
                'type': 'storage',
                'description': 'Distributed consensus for schema changes with evolution history.',
                'interfaces': ['evolution_consensus', 'migration', 'history_tracking'],
                'layer': 'storage'
            },
            'stream_processor': {
                'name': 'Stream Processor',
                'pos': (1200, 500),
                'size': (110, 60),
                'type': 'storage',
                'description': 'Real-time data stream processing with windowing and aggregation.',
                'interfaces': ['stream_processing', 'windowing', 'aggregation'],
                'layer': 'storage'
            },
            
            # Emergence Layer
            'self_organization': {
                'name': 'Self-Organization',
                'pos': (600, 800),
                'size': (130, 80),
                'type': 'emergence',
                'description': 'Network formation, load balancing, collaborative problem-solving, adaptive reconfiguration.',
                'interfaces': ['network_formation', 'collaborative_solving', 'adaptive_reconfiguration'],
                'layer': 'emergence'
            },
            'interaction_patterns': {
                'name': 'Interaction Patterns',
                'pos': (380, 800),
                'size': (140, 80),
                'type': 'emergence',
                'description': 'Gossip propagation, swarm consensus, hierarchical coordination, dyad formation.',
                'interfaces': ['gossip_propagation', 'swarm_consensus', 'hierarchical_coordination'],
                'layer': 'emergence'
            },
            'emergence_detection': {
                'name': 'Emergence Detection',
                'pos': (150, 800),
                'size': (130, 80),
                'type': 'emergence',
                'description': 'Detection and analysis of emergent behaviors and system properties.',
                'interfaces': ['emergence_detection', 'behavior_analysis', 'property_analysis'],
                'layer': 'emergence'
            }
        }
    
    def initialize_connections(self):
        """Define data flows and interactions between components"""
        return [
            # Core system connections
            ('object_core', 'meta_dsl', 'bidirectional', 'Self-modification requests'),
            ('object_core', 'system_orchestrator', 'bidirectional', 'State updates and orchestration'),
            ('meta_dsl', 'system_orchestrator', 'unidirectional', 'Evolution proposals'),
            
            # Agent connections to core
            ('object_core', 'ai_agent', 'bidirectional', 'Agent instantiation and control'),
            ('object_core', 'coordinator_object', 'bidirectional', 'Coordination requests'),
            ('object_core', 'sensor_object', 'bidirectional', 'Sensor data flow'),
            ('object_core', 'actuator_object', 'bidirectional', 'Actuation commands'),
            ('object_core', 'human_client', 'bidirectional', 'Human interaction'),
            
            # Communication layer
            ('object_core', 'mailbox_system', 'bidirectional', 'Message handling'),
            ('mailbox_system', 'message_router', 'unidirectional', 'Message routing'),
            ('message_router', 'network_transport', 'unidirectional', 'Network delivery'),
            ('network_transport', 'p2p_bootstrap', 'bidirectional', 'Peer discovery'),
            
            # Learning connections
            ('ai_agent', 'oorl_framework', 'bidirectional', 'Learning updates'),
            ('coordinator_object', 'collective_learning', 'bidirectional', 'Coalition learning'),
            ('oorl_framework', 'distributed_training', 'bidirectional', 'Distributed optimization'),
            ('collective_learning', 'transfer_learning', 'bidirectional', 'Knowledge transfer'),
            
            # Network layer connections
            ('p2p_bootstrap', 'distributed_registry', 'unidirectional', 'Peer registration'),
            ('distributed_registry', 'network_supervisor', 'bidirectional', 'Network monitoring'),
            ('network_transport', 'network_supervisor', 'bidirectional', 'Connection status'),
            
            # Security integration
            ('network_transport', 'encryption', 'bidirectional', 'Encrypted communication'),
            ('distributed_registry', 'byzantine_fault_tolerance', 'bidirectional', 'Consensus validation'),
            ('byzantine_fault_tolerance', 'trust_manager', 'bidirectional', 'Trust computation'),
            ('encryption', 'trust_manager', 'unidirectional', 'Key verification'),
            
            # Monitoring connections
            ('system_orchestrator', 'performance_monitor', 'bidirectional', 'System metrics'),
            ('performance_monitor', 'resource_manager', 'unidirectional', 'Resource allocation'),
            ('object_core', 'agent_monitor', 'unidirectional', 'Agent health data'),
            ('resource_manager', 'agent_monitor', 'bidirectional', 'Resource usage'),
            
            # Schema management
            ('meta_dsl', 'schema_registry', 'bidirectional', 'Schema definitions'),
            ('schema_registry', 'schema_evolution', 'bidirectional', 'Evolution management'),
            ('schema_evolution', 'distributed_registry', 'unidirectional', 'Schema consensus'),
            ('message_router', 'stream_processor', 'unidirectional', 'Message streams'),
            
            # Emergence connections
            ('collective_learning', 'interaction_patterns', 'bidirectional', 'Learning patterns'),
            ('interaction_patterns', 'self_organization', 'bidirectional', 'Organization patterns'),
            ('self_organization', 'emergence_detection', 'unidirectional', 'Emergence analysis'),
            ('performance_monitor', 'emergence_detection', 'unidirectional', 'Performance patterns'),
            ('system_orchestrator', 'self_organization', 'bidirectional', 'Self-organization control'),
            
            # Cross-layer connections
            ('oorl_framework', 'performance_monitor', 'unidirectional', 'Learning performance'),
            ('byzantine_fault_tolerance', 'distributed_training', 'unidirectional', 'Secure training'),
            ('trust_manager', 'collective_learning', 'unidirectional', 'Trust-based learning'),
            ('emergence_detection', 'schema_evolution', 'unidirectional', 'Emergent schema changes')
        ]
    
    def draw_diagram(self):
        """Draw the complete system diagram"""
        self.canvas.delete("all")
        
        # Draw connections first (so they appear behind components)
        self.draw_connections()
        
        # Draw components
        self.component_objects = {}
        for comp_id, comp_data in self.components.items():
            if self.layer_vars[comp_data['layer']].get():
                self.draw_component(comp_id, comp_data)
        
        # Draw legend
        self.draw_legend()
        
        # Draw title
        self.canvas.create_text(700, 50, text="AAOS - Autonomous Agency Operating System", 
                              font=('Arial', 18, 'bold'), fill='white', anchor='center')
        self.canvas.create_text(700, 75, text="Comprehensive System Architecture with Learning, Security, and Emergence", 
                              font=('Arial', 12), fill='#cccccc', anchor='center')
    
    def draw_component(self, comp_id, comp_data):
        """Draw a single component with rounded rectangle"""
        x, y = comp_data['pos']
        w, h = comp_data['size']
        color = self.colors[comp_data['type']]
        
        # Create rounded rectangle
        points = self.rounded_rectangle_points(x, y, x+w, y+h, 10)
        rect_id = self.canvas.create_polygon(points, fill=color, outline='white', width=2,
                                           tags=(comp_id, 'component'))
        
        # Component name
        text_id = self.canvas.create_text(x + w/2, y + h/2, text=comp_data['name'], 
                                        font=('Arial', 9, 'bold'), fill='white', 
                                        anchor='center', tags=(comp_id, 'component'))
        
        # Store component objects for interaction
        self.component_objects[comp_id] = {
            'rect': rect_id,
            'text': text_id,
            'data': comp_data
        }
    
    def draw_connections(self):
        """Draw all connections between components"""
        for start_id, end_id, direction, label in self.connections:
            if (start_id in self.components and end_id in self.components and
                self.layer_vars[self.components[start_id]['layer']].get() and
                self.layer_vars[self.components[end_id]['layer']].get()):
                
                start_comp = self.components[start_id]
                end_comp = self.components[end_id]
                
                start_x, start_y = start_comp['pos']
                start_w, start_h = start_comp['size']
                end_x, end_y = end_comp['pos']
                end_w, end_h = end_comp['size']
                
                # Calculate connection points (center to center)
                x1 = start_x + start_w/2
                y1 = start_y + start_h/2
                x2 = end_x + end_w/2
                y2 = end_y + end_h/2
                
                # Draw arrow
                arrow_style = tk.LAST if direction in ['unidirectional', 'bidirectional'] else None
                line_id = self.canvas.create_line(x1, y1, x2, y2, 
                                                fill='#666666', width=2, 
                                                arrow=arrow_style, arrowshape=(12, 15, 4),
                                                tags='connection')
                
                # Draw bidirectional arrow if needed
                if direction == 'bidirectional':
                    self.canvas.create_line(x1, y1, x2, y2, 
                                          fill='#666666', width=2, 
                                          arrow=tk.FIRST, arrowshape=(12, 15, 4),
                                          tags='connection')
    
    def draw_legend(self):
        """Draw legend showing component types"""
        legend_x = 50
        legend_y = 950
        
        self.canvas.create_text(legend_x, legend_y, text="Component Types:", 
                              font=('Arial', 12, 'bold'), fill='white', anchor='w')
        
        legend_items = [
            ('core', 'Core System'),
            ('agent', 'Agent Types'),
            ('communication', 'Communication'),
            ('learning', 'Learning'),
            ('network', 'Network'),
            ('security', 'Security'),
            ('monitoring', 'Monitoring'),
            ('storage', 'Storage'),
            ('emergence', 'Emergence'),
            ('control', 'Control')
        ]
        
        for i, (comp_type, label) in enumerate(legend_items):
            x = legend_x + (i % 5) * 200
            y = legend_y + 30 + (i // 5) * 25
            
            self.canvas.create_rectangle(x, y, x+15, y+15, 
                                       fill=self.colors[comp_type], outline='white')
            self.canvas.create_text(x+25, y+7, text=label, 
                                  font=('Arial', 10), fill='white', anchor='w')
    
    def rounded_rectangle_points(self, x1, y1, x2, y2, radius):
        """Generate points for a rounded rectangle"""
        points = []
        
        # Top side
        points.extend([x1 + radius, y1])
        points.extend([x2 - radius, y1])
        
        # Top-right corner
        for i in range(5):
            angle = i * math.pi / 8
            px = x2 - radius + radius * math.cos(angle)
            py = y1 + radius - radius * math.sin(angle)
            points.extend([px, py])
        
        # Right side
        points.extend([x2, y1 + radius])
        points.extend([x2, y2 - radius])
        
        # Bottom-right corner
        for i in range(5):
            angle = i * math.pi / 8
            px = x2 - radius + radius * math.sin(angle)
            py = y2 - radius + radius * math.cos(angle)
            points.extend([px, py])
        
        # Bottom side
        points.extend([x2 - radius, y2])
        points.extend([x1 + radius, y2])
        
        # Bottom-left corner
        for i in range(5):
            angle = i * math.pi / 8
            px = x1 + radius - radius * math.cos(angle)
            py = y2 - radius + radius * math.sin(angle)
            points.extend([px, py])
        
        # Left side
        points.extend([x1, y2 - radius])
        points.extend([x1, y1 + radius])
        
        # Top-left corner
        for i in range(5):
            angle = i * math.pi / 8
            px = x1 + radius - radius * math.sin(angle)
            py = y1 + radius - radius * math.cos(angle)
            points.extend([px, py])
        
        return points
    
    def update_visibility(self):
        """Update component visibility based on layer toggles"""
        self.draw_diagram()
    
    def on_click(self, event):
        """Handle mouse click events"""
        # Convert canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Find clicked component
        clicked_items = self.canvas.find_overlapping(x-5, y-5, x+5, y+5)
        
        for item in clicked_items:
            tags = self.canvas.gettags(item)
            
            for tag in tags:
                if tag in self.component_objects:
                    self.selected_component = tag
                    self.show_component_info(tag)
                    self.drag_data["x"] = x
                    self.drag_data["y"] = y
                    return
        
        # Clear selection if clicking empty space
        self.selected_component = None
        self.info_text.delete(1.0, tk.END)
    
    def on_drag(self, event):
        """Handle mouse drag events"""
        if self.selected_component:
            # Convert canvas coordinates
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            
            # Calculate delta
            dx = x - self.drag_data["x"]
            dy = y - self.drag_data["y"]
            
            # Move component
            self.canvas.move(self.selected_component, dx, dy)
            
            # Update stored position
            comp_data = self.components[self.selected_component]
            old_x, old_y = comp_data['pos']
            comp_data['pos'] = (old_x + dx, old_y + dy)
            
            # Update drag data
            self.drag_data["x"] = x
            self.drag_data["y"] = y
            
            # Redraw connections
            self.canvas.delete('connection')
            self.draw_connections()
    
    def on_release(self, event):
        """Handle mouse release events"""
        pass
    
    def on_mousewheel(self, event):
        """Handle mouse wheel for zooming"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def show_component_info(self, comp_id):
        """Display detailed information about a component"""
        comp_data = self.components[comp_id]
        
        info = f"Component: {comp_data['name']}\n"
        info += f"Type: {comp_data['type'].title()}\n"
        info += f"Layer: {comp_data['layer'].title()}\n\n"
        info += f"Description:\n{comp_data['description']}\n\n"
        info += f"Interfaces:\n"
        
        for interface in comp_data['interfaces']:
            info += f"  • {interface.replace('_', ' ').title()}\n"
        
        # Add connection information
        connections_in = []
        connections_out = []
        
        for start_id, end_id, direction, label in self.connections:
            if start_id == comp_id:
                connections_out.append(f"{end_id}: {label}")
            elif end_id == comp_id:
                connections_in.append(f"{start_id}: {label}")
        
        if connections_in:
            info += f"\nIncoming Connections:\n"
            for conn in connections_in[:5]:  # Limit display
                info += f"  ← {conn}\n"
        
        if connections_out:
            info += f"\nOutgoing Connections:\n"
            for conn in connections_out[:5]:  # Limit display
                info += f"  → {conn}\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)

def main():
    root = tk.Tk()
    app = AAOSSystemDiagram(root)
    
    # Add menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Export Diagram", command=lambda: messagebox.showinfo("Export", "Export functionality would be implemented here"))
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    view_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="View", menu=view_menu)
    view_menu.add_command(label="Reset View", command=app.draw_diagram)
    view_menu.add_command(label="Show All Layers", command=lambda: [var.set(True) for var in app.layer_vars.values()] or app.update_visibility())
    view_menu.add_command(label="Hide All Layers", command=lambda: [var.set(False) for var in app.layer_vars.values()] or app.update_visibility())
    
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", 
        "AAOS System Diagram\n\nInteractive visualization of the Autonomous Agency Operating System architecture.\n\n"
        "Features:\n• Click components for detailed information\n• Drag components to reposition\n• Toggle layers on/off\n• Explore system relationships"))
    
    root.mainloop()

if __name__ == "__main__":
    main()