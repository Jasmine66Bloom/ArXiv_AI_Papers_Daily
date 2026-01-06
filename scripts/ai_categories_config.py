"""
AI/算法领域论文分类配置文件 (2026前瞻版)
基于当前AI领域的实际热点和发展脉络，覆盖短期爆发方向和长期基础方向
"""

# 类别阈值配置（越大越严格）
CATEGORY_THRESHOLDS = {
    # 1. AI Agents & Reasoning（智能体与推理）
    "AI Agents & Reasoning (智能体与推理)": {
        "threshold": 0.7,
        "subcategories": {
            "单智能体规划 (Single-Agent Planning)": 0.7,
            "多智能体协作 (Multi-Agent Collaboration)": 0.7,
            "链式推理 (Chain-of-Thought Reasoning)": 0.7,
            "上下文管理 (Context Management)": 0.7,
            "智能体评估 (Agent Evaluation)": 0.7,
            "工作流编排 (Workflow Orchestration)": 0.7,
        },
        "priority": 5.5
    },

    # 2. Multimodal Models（多模态模型）
    "Multimodal Models (多模态模型)": {
        "threshold": 0.7,
        "subcategories": {
            "视觉-语言模型 (Vision-Language Models)": 0.7,
            "视频理解 (Video Understanding)": 0.7,
            "音频-视觉融合 (Audio-Visual Fusion)": 0.7,
            "3D重建 (3D Reconstruction)": 0.7,
            "多模态生成 (Multimodal Generation)": 0.7,
            "统一预训练 (Unified Pretraining)": 0.7,
        },
        "priority": 5.3
    },

    # 3. Efficient Large Models（大模型高效训练与推理）
    "Efficient Large Models (大模型高效训练与推理)": {
        "threshold": 0.7,
        "subcategories": {
            "模型压缩 (Model Compression)": 0.7,
            "推理优化 (Inference Optimization)": 0.7,
            "混合专家 (Mixture of Experts)": 0.7,
            "合成数据 (Synthetic Data)": 0.7,
            "绿色计算 (Green Computing)": 0.7,
        },
        "priority": 5.0
    },

    # 4. Embodied AI & Robotics（具身智能与机器人）
    "Embodied AI & Robotics (具身智能与机器人)": {
        "threshold": 0.7,
        "subcategories": {
            "机器人学习 (Robot Learning)": 0.7,
            "仿真迁移 (Sim-to-Real Transfer)": 0.7,
            "世界模型 (World Models)": 0.7,
            "机器人基础模型 (Foundation Models for Robotics)": 0.7,
            "运动规划 (Motion Planning)": 0.7,
        },
        "priority": 5.2
    },

    # 5. AI Safety, Alignment & Interpretability（安全、对齐与可解释性）
    "AI Safety, Alignment & Interpretability (安全、对齐与可解释性)": {
        "threshold": 0.7,
        "subcategories": {
            "价值对齐 (Value Alignment)": 0.7,
            "机制可解释性 (Mechanistic Interpretability)": 0.7,
            "幻觉检测 (Hallucination Detection)": 0.7,
            "安全评估 (Safety Evaluation)": 0.7,
            "隐私保护 (Privacy Protection)": 0.7,
        },
        "priority": 5.1
    },

    # 6. Domain-Specific & Personalized AI（垂直领域与个性化）
    "Domain-Specific & Personalized AI (垂直领域与个性化)": {
        "threshold": 0.7,
        "subcategories": {
            "个性化模型 (Personalized Models)": 0.7,
            "联邦学习 (Federated Learning)": 0.7,
            "科学计算 (Scientific Computing)": 0.7,
            "医疗AI (Medical AI)": 0.7,
            "金融与法律AI (Financial & Legal AI)": 0.7,
        },
        "priority": 4.9
    }
}

# 类别显示顺序配置
CATEGORY_DISPLAY_ORDER = [
    # 1. AI Agents & Reasoning（智能体与推理）
    "AI Agents & Reasoning (智能体与推理)",

    # 2. Multimodal Models（多模态模型）
    "Multimodal Models (多模态模型)",

    # 3. Efficient Large Models（大模型高效训练与推理）
    "Efficient Large Models (大模型高效训练与推理)",

    # 4. Embodied AI & Robotics（具身智能与机器人）
    "Embodied AI & Robotics (具身智能与机器人)",

    # 5. AI Safety, Alignment & Interpretability（安全、对齐与可解释性）
    "AI Safety, Alignment & Interpretability (安全、对齐与可解释性)",

    # 6. Domain-Specific & Personalized AI（垂直领域与个性化）
    "Domain-Specific & Personalized AI (垂直领域与个性化)"
]

# 分类提示词
CATEGORY_PROMPT = """
请将以下AI/算法论文分类到最合适的类别中。

2026年AI/算法领域分类体系：

# 1. AI Agents & Reasoning（智能体与推理）
"AI Agents & Reasoning (智能体与推理)"：单智能体规划、多智能体协作、链式推理、上下文管理、智能体评估、工作流编排
   定义：研究自主智能体的架构、协作机制、规划能力、推理能力和工作流编排等。

# 2. Multimodal Models（多模态模型）
"Multimodal Models (多模态模型)"：视觉-语言模型、视频理解、音频-视觉融合、3D重建、多模态生成、统一预训练
   定义：研究跨模态理解和生成的模型，包括视觉-语言、视频、音频、3D等多模态融合和生成。

# 3. Efficient Large Models（大模型高效训练与推理）
"Efficient Large Models (大模型高效训练与推理)"：模型压缩、推理优化、混合专家、合成数据、绿色计算
   定义：研究大模型的高效训练和推理方法，包括模型压缩、推理优化、MoE、数据合成和绿色计算等技术。

# 4. Embodied AI & Robotics（具身智能与机器人）
"Embodied AI & Robotics (具身智能与机器人)"：机器人学习、仿真迁移、世界模型、机器人基础模型、运动规划
   定义：研究具身智能和机器人学习，包括机器人学习、仿真迁移、世界模型、基础模型应用和运动规划等。

# 5. AI Safety, Alignment & Interpretability（安全、对齐与可解释性）
"AI Safety, Alignment & Interpretability (安全、对齐与可解释性)"：价值对齐、机制可解释性、幻觉检测、安全评估、隐私保护
   定义：研究AI的安全性、对齐方法和可解释性，包括价值对齐、机制解释、幻觉检测、安全评估和隐私保护等。

# 6. Domain-Specific & Personalized AI（垂直领域与个性化）
"Domain-Specific & Personalized AI (垂直领域与个性化)"：个性化模型、联邦学习、科学计算、医疗AI、金融与法律AI
   定义：研究个性化AI和垂直领域应用，包括个性化模型、联邦学习、科学计算、医疗、金融和法律等领域的AI应用。

分类指南：
1. 首先分析论文的核心技术贡献和主要研究目标
2. 考虑论文的方法、实验和应用场景
3. 如果论文涉及多个类别，请选择最核心、最具创新性的方向
4. 优先考虑技术本质而非应用领域（除非应用创新是论文的主要贡献）

边界案例处理：
- 如果论文同时涉及"多模态模型"和"AI Agents"，但核心是多模态理解，应归类为"多模态模型"
- 如果论文研究机器人技术，即使涉及多模态，也应优先归类为"具身智能与机器人"而非"多模态模型"
- 如果论文提出新的推理方法，应归类为"AI Agents & Reasoning"而非其他类别
- 如果论文研究大模型效率优化，应归类为"大模型高效训练与推理"而非其他类别

请分析论文的核心技术和主要贡献，选择最合适的一个类别。只返回类别名称，不要有任何解释或额外文本。
"""

# 类别关键词配置
CATEGORY_KEYWORDS = {
    # 1. AI Agents & Reasoning（智能体与推理）
    "AI Agents & Reasoning (智能体与推理)": {
        "keywords": [
            # 单代理规划与工具使用（高权重）
            ("autonomous agent", 2.5),
            ("tool use", 2.5),
            ("ReAct", 2.5),
            ("reflexion", 2.5),
            ("agent planning", 2.5),
            ("tool-using", 2.3),
            ("function calling", 2.3),
            ("API calling", 2.3),
            ("tool calling", 2.3),
            ("agent trajectory", 2.3),
            ("self-refine", 2.3),
            ("LLM agent", 2.3),

            # 多代理协作系统（高权重）
            ("multi-agent system", 2.5),
            ("multi-agent collaboration", 2.5),
            ("agent society", 2.5),
            ("emergent behavior", 2.5),
            ("multi-agent", 2.3),
            ("agent interaction", 2.3),
            ("agent cooperation", 2.3),
            ("agent coordination", 2.3),
            ("multi-agent debate", 2.3),
            ("agent swarm", 2.3),
            ("cooperative agents", 2.3),
            ("competitive agents", 2.3),
            ("MAS framework", 2.3),
            ("decentralized agents", 2.3),

            # 长链推理与思考链（高权重）
            ("chain-of-thought", 2.5),
            ("o1-like", 2.5),
            ("test-time compute", 2.5),
            ("long-chain reasoning", 2.5),
            ("reasoning model", 2.5),
            ("CoT", 2.3),
            ("reasoning capability", 2.3),
            ("complex reasoning", 2.3),
            ("logical reasoning", 2.3),
            ("step-by-step reasoning", 2.3),
            ("tree of thoughts", 2.3),
            ("graph of thoughts", 2.3),
            ("reasoning engine", 2.3),
            ("scaled inference", 2.3),

            # 上下文工程（高权重）
            ("context engineering", 2.5),
            ("agentic context", 2.5),
            ("dynamic context", 2.5),
            ("context optimization", 2.5),
            ("in-context management", 2.5),
            ("context management", 2.3),
            ("context compression", 2.3),
            ("RAG", 2.3),
            ("RAG optimization", 2.3),
            ("long context management", 2.3),
            ("context window extension", 2.3),
            ("prompt context design", 2.3),
            ("adaptive context", 2.3),

            # Agent评估与基准（高权重）
            ("agent benchmark", 2.5),
            ("agent evaluation", 2.5),
            ("GAIA", 2.5),
            ("WebArena", 2.5),
            ("ToolBench", 2.3),
            ("AgentBoard", 2.3),
            ("Berkeley Function Calling Leaderboard", 2.3),
            ("agent performance", 2.3),
            ("agent testing", 2.3),
            ("agent leaderboard", 2.3),
            ("agent capability assessment", 2.3),
            ("benchmark suite agents", 2.3),

            # Agentic Workflow与自动化（高权重）
            ("agentic workflow", 2.5),
            ("agent orchestration", 2.5),
            ("long-horizon task", 2.5),
            ("task decomposition", 2.3),
            ("workflow automation", 2.3),
            ("autonomous execution", 2.3),
            ("agent pipeline", 2.3),
            ("multi-step agent", 2.3),
            ("hierarchical agents", 2.3),
            ("workflow automation AI", 2.3),
            ("long-term planning agents", 2.3),

            # 其他Agent相关
            ("AI agent", 2.5),
            ("agentic", 2.5),
            ("agent-based", 2.3),
            ("self-reflection", 2.3),
            ("agent memory", 2.3),
            ("long-term memory", 2.3),
        ],
        "negative_keywords": [
            ("computer vision", 1.0),
            ("image classification", 1.0),
            ("object detection", 1.0),
        ]
    },

    # 2. Multimodal Models（多模态模型）
    "Multimodal Models (多模态模型)": {
        "keywords": [
            # 视觉-语言模型（VLM）（高权重）
            ("vision-language model", 2.5),
            ("VLM", 2.5),
            ("image-text alignment", 2.5),
            ("visual question answering", 2.5),
            ("image captioning", 2.5),
            ("CLIP variant", 2.5),
            ("visual grounding", 2.5),
            ("multimodal alignment", 2.5),
            ("vision encoder", 2.5),
            ("LLM vision", 2.5),
            ("vision-language", 2.3),
            ("visual-language", 2.3),
            ("multimodal transformer", 2.5),

            # 视频与时序多模态（高权重）
            ("video understanding", 2.5),
            ("video-language", 2.5),
            ("temporal multimodal", 2.5),
            ("long video model", 2.5),
            ("video action recognition", 2.5),
            ("video temporal modeling", 2.5),
            ("spatio-temporal", 2.5),
            ("video frame interpolation", 2.5),
            ("video reasoning", 2.5),
            ("dynamic scene understanding", 2.5),
            ("video captioning", 2.3),
            ("video-text", 2.3),
            ("video generation", 2.3),
            ("action recognition", 2.3),

            # 音频-视觉-文本融合（高权重）
            ("audio-visual", 2.5),
            ("speech multimodal", 2.5),
            ("audio-language model", 2.5),
            ("audio captioning", 2.5),
            ("speech-to-text multimodal", 2.5),
            ("audio event detection", 2.5),
            ("music understanding", 2.5),
            ("AV alignment", 2.5),
            ("sound localization", 2.5),
            ("multimodal audio processing", 2.5),
            ("audio-text", 2.3),
            ("speech-language", 2.3),
            ("audio event", 2.3),
            ("music generation", 2.3),

            # 3D/4D与空间多模态（高权重）
            ("3D multimodal", 2.5),
            ("4D generation", 2.5),
            ("gaussian splatting", 2.5),
            ("spatial understanding", 2.5),
            ("NeRF multimodal", 2.5),
            ("3D reconstruction", 2.5),
            ("point cloud multimodal", 2.5),
            ("scene graph 3D", 2.5),
            ("4D video", 2.5),
            ("dynamic 3D modeling", 2.5),
            ("neural rendering", 2.3),
            ("NeRF", 2.3),
            ("3D vision", 2.3),

            # 生成式多模态（高权重）
            ("generative multimodal", 2.5),
            ("diffusion model multimodal", 2.5),
            ("generative AI", 2.5),
            ("AIGC", 2.5),
            ("multimodal generation", 2.5),
            ("video diffusion", 2.5),
            ("image diffusion multimodal", 2.5),
            ("audio diffusion", 2.5),
            ("3D diffusion", 2.5),
            ("text-to-multimodal", 2.5),
            ("Sora-like", 2.5),
            ("text-to-image", 2.3),
            ("text-to-video", 2.3),
            ("image generation", 2.3),

            # 统一多模态预训练（高权重）
            ("unified multimodal", 2.5),
            ("any-to-any", 2.5),
            ("multimodal foundation model", 2.5),
            ("unified encoder", 2.5),
            ("cross-modal pretraining", 2.5),
            ("omni-modal", 2.5),
            ("single transformer multimodal", 2.5),
            ("modality agnostic", 2.5),
            ("universal representation", 2.5),
            ("Chameleon-like", 2.5),
            ("omnimodal", 2.3),
            ("multimodal pretraining", 2.3),

            # 其他多模态相关
            ("multimodal", 2.0),
            ("cross-modal", 2.5),
            ("cross-modal alignment", 2.5),
            ("modality alignment", 2.3),
            ("multimodal alignment", 2.3),
            ("interleaved modal", 2.3),
            ("MMDiT", 2.5),
            ("multimodal diffusion", 2.3),
        ],
        "negative_keywords": [
            ("single-modal", 1.2),
            ("unimodal", 1.2),
            ("text-only", 1.0),
        ]
    },

    # 3. Efficient Large Models（大模型高效训练与推理）
    "Efficient Large Models (大模型高效训练与推理)": {
        "keywords": [
            # 模型压缩与量化（高权重）
            ("model compression", 2.5),
            ("quantization", 2.5),
            ("pruning", 2.5),
            ("LoRA", 2.5),
            ("QLoRA", 2.5),
            ("low-rank adaptation", 2.5),
            ("parameter efficient", 2.5),
            ("structured pruning", 2.5),
            ("weight sharing", 2.5),
            ("bit quantization", 2.5),
            ("post-training quantization", 2.5),
            ("model quantization", 2.5),
            ("quantization-aware training", 2.3),
            ("network pruning", 2.3),
            ("model pruning", 2.3),
            ("low-rank adaptation", 2.3),

            # 推理加速技术（高权重）
            ("inference optimization", 2.5),
            ("speculative decoding", 2.5),
            ("flash attention", 2.5),
            ("KV cache", 2.5),
            ("medusa decoding", 2.5),
            ("paged attention", 2.5),
            ("continuous batching", 2.5),
            ("attention optimization", 2.5),
            ("inference engine", 2.5),
            ("vLLM", 2.5),
            ("inference acceleration", 2.5),
            ("fast inference", 2.3),
            ("inference speedup", 2.3),
            ("KV cache optimization", 2.3),

            # 混合专家模型（MoE）（高权重）
            ("mixture of experts", 2.5),
            ("MoE", 2.5),
            ("sparse MoE", 2.5),
            ("expert routing", 2.5),
            ("switch transformer", 2.5),
            ("MoE scaling", 2.5),
            ("dynamic routing", 2.5),
            ("expert choice", 2.5),
            ("top-k routing", 2.5),
            ("MoE inference", 2.5),
            ("sparse expert", 2.3),
            ("mixture-of-experts", 2.3),

            # 合成数据生成（高权重）
            ("synthetic data generation", 2.5),
            ("data distillation", 2.5),
            ("self-reward data", 2.5),
            ("synthetic training data", 2.5),
            ("data mixture", 2.5),
            ("evolutionary data", 2.5),
            ("self-improvement data", 2.5),
            ("reward modeling data", 2.5),
            ("preference data synthesis", 2.5),
            ("distilled dataset", 2.5),
            ("synthetic data", 2.3),
            ("data augmentation", 2.0),
            ("self-supervised data", 2.0),

            # 能效与可持续训练（高权重）
            ("energy efficient AI", 2.5),
            ("green AI", 2.5),
            ("carbon aware computing", 2.5),
            ("hardware-aware training", 2.5),
            ("sustainable LLM", 2.5),
            ("power optimization", 2.5),
            ("training efficiency", 2.5),
            ("flop reduction", 2.5),
            ("eco-friendly AI", 2.5),
            ("compute aware scheduling", 2.5),
            ("low-power training", 2.3),
            ("energy optimization", 2.3),
            ("sustainable training", 2.3),

            # 其他效率相关
            ("efficient training", 2.3),
            ("parameter-efficient", 2.3),
            ("PEFT", 2.3),
            ("adapter", 2.0),
        ],
        "negative_keywords": [
            ("accuracy improvement", 0.8),
            ("performance gain", 0.8),
        ]
    },

    # 4. Embodied AI & Robotics（具身智能与机器人）
    "Embodied AI & Robotics (具身智能与机器人)": {
        "keywords": [
            # 机器人学习基础（高权重）
            ("robot learning", 2.5),
            ("reinforcement learning robotics", 2.5),
            ("imitation learning robot", 2.5),
            ("behavior cloning robot", 2.5),
            ("RL fine-tuning robot", 2.5),
            ("policy learning", 2.5),
            ("robotic control RL", 2.5),
            ("offline RL robot", 2.5),
            ("multi-task robot learning", 2.5),
            ("robotic learning", 2.3),
            ("robotics", 2.3),
            ("robot control", 2.3),
            ("policy learning", 2.3),

            # 仿真到现实迁移（Sim-to-Real）（高权重）
            ("sim-to-real", 2.5),
            ("domain randomization", 2.5),
            ("sim2real transfer", 2.5),
            ("reality gap", 2.5),
            ("domain adaptation robot", 2.5),
            ("simulation optimization", 2.5),
            ("physics randomization", 2.5),
            ("zero-shot sim2real", 2.5),
            ("affordance transfer", 2.5),
            ("simulation to reality", 2.3),
            ("domain adaptation robotics", 2.3),
            ("real-world robot", 2.3),
            ("sim2real", 2.3),

            # 世界模型与预测（高权重）
            ("world model", 2.5),
            ("video prediction robotics", 2.5),
            ("physical reasoning", 2.5),
            ("generative world model", 2.5),
            ("dynamics modeling", 2.5),
            ("state space model robot", 2.5),
            ("predictive modeling embodied", 2.5),
            ("environment simulation", 2.5),
            ("Gaussian world model", 2.5),
            ("world modeling", 2.3),
            ("environment model", 2.3),
            ("physics-based", 2.3),

            # 基础模型在机器人（高权重）
            ("foundation model robotics", 2.5),
            ("large model robotics", 2.5),
            ("RT-X", 2.5),
            ("robotic foundation model", 2.5),
            ("pretrained robot policy", 2.5),
            ("vision-language-action", 2.5),
            ("VLA model", 2.5),
            ("embodied foundation", 2.5),
            ("open-x embodiment", 2.5),
            ("embodied foundation model", 2.3),
            ("robot foundation model", 2.3),
            ("embodied pretraining", 2.3),

            # 灵巧操作与人形机器人（高权重）
            ("dexterous manipulation", 2.5),
            ("humanoid robot", 2.5),
            ("bipedal locomotion", 2.5),
            ("multi-finger grasp", 2.5),
            ("in-hand manipulation", 2.5),
            ("humanoid control", 2.5),
            ("legged robot", 2.5),
            ("anthropomorphic hand", 2.5),
            ("tactile sensing robot", 2.5),
            ("grasping", 2.3),
            ("manipulation", 2.3),
            ("hand manipulation", 2.3),
            ("multi-finger", 2.3),

            # 其他具身智能相关
            ("embodied AI", 2.5),
            ("embodied intelligence", 2.5),
            ("embodied", 2.3),
            ("navigation", 2.3),
            ("dexterity", 2.3),
            ("robotic vision", 2.0),
        ],
        "negative_keywords": [
            ("pure vision", 1.0),
            ("image processing", 1.0),
            ("computer vision only", 1.2),
        ]
    },

    # 5. AI Safety, Alignment & Interpretability（安全、对齐与可解释性）
    "AI Safety, Alignment & Interpretability (安全、对齐与可解释性)": {
        "keywords": [
            # 价值对齐与宪法AI（高权重）
            ("AI alignment", 2.5),
            ("RLHF", 2.5),
            ("constitutional AI", 2.5),
            ("preference optimization", 2.5),
            ("DPO", 2.5),
            ("PPO alignment", 2.5),
            ("reward modeling", 2.5),
            ("human feedback", 2.5),
            ("value alignment", 2.5),
            ("super alignment", 2.5),
            ("value alignment", 2.3),
            ("RLAIF", 2.3),
            ("human feedback", 2.3),
            ("DPO", 2.3),

            # 机制可解释性（高权重）
            ("mechanistic interpretability", 2.5),
            ("circuit discovery", 2.5),
            ("superposition", 2.5),
            ("neuron interpretation", 2.5),
            ("feature visualization", 2.5),
            ("activation patching", 2.5),
            ("dictionary learning", 2.5),
            ("sparse autoencoder", 2.5),
            ("interpretability circuits", 2.5),
            ("interpretability", 2.3),
            ("model interpretability", 2.3),
            ("explainable AI", 2.3),
            ("XAI", 2.3),
            ("feature analysis", 2.3),

            # 幻觉与鲁棒性（高权重）
            ("hallucination mitigation", 2.5),
            ("adversarial robustness", 2.5),
            ("out-of-distribution", 2.5),
            ("factuality improvement", 2.5),
            ("uncertainty estimation", 2.5),
            ("robust training", 2.5),
            ("OOD detection", 2.5),
            ("hallucination benchmark", 2.5),
            ("confidence calibration", 2.5),
            ("hallucination reduction", 2.3),
            ("factual accuracy", 2.3),
            ("factuality", 2.3),
            ("adversarial attack", 2.3),
            ("robustness", 2.3),
            ("adversarial defense", 2.3),

            # 红队测试与安全评估（高权重）
            ("red teaming", 2.5),
            ("jailbreak", 2.5),
            ("AI safety benchmark", 2.5),
            ("safety evaluation", 2.5),
            ("capability elicitation", 2.5),
            ("vulnerability probing", 2.5),
            ("attack simulation", 2.5),
            ("safety fine-tuning", 2.5),
            ("refusal mechanism", 2.5),
            ("red team", 2.3),
            ("adversarial testing", 2.3),
            ("safety testing", 2.3),
            ("safety evaluation", 2.3),

            # 隐私与公平性（高权重）
            ("differential privacy AI", 2.5),
            ("bias mitigation", 2.5),
            ("poisoning attack", 2.5),
            ("fairness AI", 2.5),
            ("demographic parity", 2.5),
            ("backdoor defense", 2.5),
            ("privacy preserving", 2.5),
            ("unlearning", 2.5),
            ("equity modeling", 2.5),
            ("privacy preservation", 2.3),
            ("differential privacy", 2.3),
            ("privacy-preserving", 2.3),
            ("data privacy", 2.0),
            ("fairness", 2.3),

            # 其他安全相关
            ("AI safety", 2.5),
            ("safety alignment", 2.3),
            ("harmful content", 2.0),
            ("safety mechanism", 2.0),
        ],
        "negative_keywords": [
            ("performance optimization", 0.8),
            ("accuracy improvement", 0.8),
        ]
    },

    # 6. Domain-Specific & Personalized AI（垂直领域与个性化）
    "Domain-Specific & Personalized AI (垂直领域与个性化)": {
        "keywords": [
            # 个性化大模型（高权重）
            ("personalized LLM", 2.5),
            ("personal AI agent", 2.5),
            ("user adaptation", 2.5),
            ("personal memory", 2.5),
            ("user profile modeling", 2.5),
            ("adaptive LLM", 2.5),
            ("customized agent", 2.5),
            ("long-term user memory", 2.5),
            ("preference tuning", 2.5),
            ("personalization", 2.3),
            ("user preference", 2.3),
            ("user modeling", 2.3),
            ("adaptive model", 2.3),

            # 联邦与隐私保护学习（高权重）
            ("federated learning", 2.5),
            ("privacy-preserving ML", 2.5),
            ("decentralized training", 2.5),
            ("federated averaging", 2.5),
            ("secure aggregation", 2.5),
            ("on-device learning", 2.5),
            ("cross-silo federated", 2.5),
            ("differential privacy federated", 2.5),
            ("federated", 2.3),
            ("distributed learning", 2.3),
            ("privacy-preserving", 2.3),

            # AI for Science（高权重）
            ("AI for science", 2.5),
            ("scientific discovery", 2.5),
            ("AlphaFold", 2.5),
            ("materials discovery", 2.5),
            ("drug design AI", 2.5),
            ("physics simulation AI", 2.5),
            ("mathematical reasoning AI", 2.5),
            ("theorem proving", 2.5),
            ("computational biology", 2.5),
            ("molecular modeling", 2.3),
            ("protein structure", 2.3),
            ("scientific ML", 2.3),
            ("physics-informed", 2.3),

            # 医疗健康AI（高权重）
            ("medical AI", 2.5),
            ("healthcare LLM", 2.5),
            ("clinical foundation model", 2.5),
            ("medical imaging AI", 2.5),
            ("drug discovery LLM", 2.5),
            ("EHR analysis", 2.5),
            ("pathology AI", 2.5),
            ("radiology foundation", 2.5),
            ("precision medicine AI", 2.5),
            ("clinical AI", 2.3),
            ("healthcare AI", 2.3),
            ("diagnosis AI", 2.3),
            ("drug discovery", 2.3),

            # 金融与法律AI（高权重）
            ("financial AI", 2.5),
            ("legal LLM", 2.5),
            ("risk modeling AI", 2.5),
            ("fraud detection LLM", 2.5),
            ("contract review AI", 2.5),
            ("quantitative finance AI", 2.5),
            ("compliance AI", 2.5),
            ("regulatory tech", 2.5),
            ("market prediction LLM", 2.5),
            ("fintech AI", 2.3),
            ("legal AI", 2.3),
            ("contract analysis", 2.3),
            ("quantitative trading", 2.3),
            ("risk assessment", 2.3),

            # 其他领域相关
            ("domain-specific", 2.3),
            ("vertical AI", 2.3),
            ("domain adaptation", 2.3),
            ("specialized model", 2.0),
        ],
        "negative_keywords": [
            ("general purpose", 0.8),
            ("broad application", 0.8),
        ]
    },
}
