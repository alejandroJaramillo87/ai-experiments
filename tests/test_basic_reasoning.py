import requests
import json
import time
import os
import textwrap

# 1. Configuration
# The URL points to your llama.cpp server's OpenAI-compatible endpoint.
# This is based on the port you exposed in your docker-compose.cpu.yml.
API_URL = "http://127.0.0.1:8001/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json"
}
# Create a directory to store the results
os.makedirs("test_results", exist_ok=True)


# 2. Test Case Definitions
# We define three distinct prompts to test different model capabilities.
# Each test is designed to push the model's context handling and generation length
# towards the 8192 token context limit.

TEST_CASES = [
    {
        "name": "Test 1: Long-Form Creative & Structured Generation",
        "prompt": """
You are a senior xenobiologist writing a highly detailed, classified report for the United Nations Extrasolar Council. Your persona is Dr. Aris Thorne, known for rigorous, data-driven, yet insightful speculation.

Generate the full report, designated File #X-77-B41, on the exoplanet Kepler-186f. The report must be exceptionally detailed, expanding on known scientific data with plausible, complex, and internally consistent inventions. The goal is to generate a very long and coherent document that adheres strictly to the specified structure.

The report must follow this exact, multi-level structure. Elaborate extensively on every single point.

**Report File #X-77-B41: Comprehensive Xenological Survey of Kepler-186f**

* **1.0 Executive Summary:** A concise overview of the planet's significance, key biological findings, the nature of the identified anomaly, and the primary recommendation for a follow-up mission.

* **2.0 Stellar & Planetary System Dynamics:**
    * **2.1** Detailed profile of the Kepler-186 red dwarf star, including its spectral type, lower luminosity, and higher proportion of red light.
    * **2.2** Analysis of Kepler-186f's orbit within the habitable zone and the profound implications of its likely tidal-locking.
    * **2.3** Climatology of the Terminator Zone: A deep analysis of the perpetual twilight band, including stable temperature gradients, prevailing wind patterns, and the "Red Dawn" phenomenon.

* **3.0 Dominant Biosphere: Flora & Fungi**
    * **3.1 The Crimson Weave (Primary Photosynthesizer):** Describe its biological structure, its use of bacteriochlorophylls to absorb red light, its colony-based structure forming vast, interconnected mats, and its reproductive cycle tied to stellar flares.
    * **3.2 Bioluminescent Fungi (Noctis Lumina):** Detail their role in the dark side ecosystem, the chemical process behind their bioluminescence (luciferin-luciferase system adapted for alien biochemistry), and their symbiotic relationship with other life.
    * **3.3 Lithophytic Flora (Stone-Eaters):** Describe organisms that derive energy from chemosynthesis, breaking down mineral substrates on the rock faces of the terminator zone.

* **4.0 Dominant Biosphere: Fauna**
    * **4.1 Primary Herbivores (The 'Glyders'):** Six-limbed, low-gravity creatures that graze on the Crimson Weave. Detail their anatomy, low-energy locomotion (gliding), and sensory organs adapted for low light.
    * **4.2 Apex Predator (The 'Stalker of Twilight'):** A solitary, ambush hunter. Detail its sophisticated camouflage (bio-chromatic skin), primary sensory organs (sonar-based echolocation), hunting strategy, and speculative social structure for mating.

* **5.0 Unexplained Anomaly: The 'Geometric Scars'**
    * **5.1 Description:** Detail the massive, perfectly straight lines and geometric patterns etched onto the planet's surface, visible from orbit. They are hundreds of kilometers long and show signs of extreme heat exposure.
    * **5.2 Competing Hypotheses:**
        * **Hypothesis A (Natural Formation):** Argue for a rare, crystalline geological faulting process amplified by tidal stresses.
        * **Hypothesis B (Technosignature):** Argue for the possibility of an extinct or dormant technological civilization, considering the unnatural precision of the scars.

* **6.0 Conclusion & Formal Recommendations**
    * **6.1 Risk Assessment:** Analyze risks for a robotic probe, including atmospheric composition, potential biological hazards, and unknown variables related to the anomaly.
    * **6.2 Primary Recommendation:** Formally recommend the deployment of the 'Odysseus' series robotic lander to the most promising terminator zone location for in-situ analysis.
""",
        "params": {
            "max_tokens": 6500,
            "stream": False,
            "top_p": 0.9,  # Use top_p for controlled creativity instead of temperature.
            "repeat_penalty": 1.15 # Add a penalty to prevent repetitive loops in long text.
        }
    },
    {
        "name": "Test 2: Complex Code Generation & Logic",
        "prompt": """
You are an expert Python architect specializing in high-performance, asynchronous networking applications.

Your task is to write a complete, production-ready Python class called `AsyncWebScraper`. This class must be designed to fetch and parse multiple web pages concurrently and safely.

**Core Requirements:**

1.  **Class Structure:** Create a class `AsyncWebScraper`. It should be initialized with a list of URLs and a `concurrency_limit` (integer).
2.  **Asynchronous Networking:** Use the `aiohttp` library for all HTTP requests to ensure non-blocking I/O.
3.  **Concurrency Control:** Use `asyncio.Semaphore` to strictly limit the number of concurrent requests to the `concurrency_limit` specified during initialization.
4.  **HTML Parsing:** Use `BeautifulSoup` to parse the HTML content of each successfully fetched page.
5.  **Error Handling:**
    * Implement robust error handling for network issues (e.g., `aiohttp.ClientError`).
    * Handle HTTP status codes gracefully (e.g., only parse `200 OK` responses).
    * The scraper should not crash if one URL fails; it should log the error and continue.
6.  **Data Extraction:** The parsing logic should extract the text from all `<h2>` tags on a page.
7.  **Main Method:** The class must have a primary public method, `run()`, which orchestrates the entire process: setting up the `aiohttp.ClientSession`, running the concurrent fetch/parse tasks, and returning the final results.
8.  **Return Value:** The `run()` method must return a list of dictionaries. Each dictionary represents a successfully scraped page and should have the structure: `{"url": "...", "h2_tags": ["tag1_text", "tag2_text", ...]}`. Failed URLs should not be in the output list.
9.  **Code Quality:** The entire response must be a single Python code block. Include all necessary imports (`asyncio`, `aiohttp`, `bs4`), full type hinting for all methods and variables, and a comprehensive docstring for the class explaining its purpose and usage.

Do not include any example usage or explanatory text outside of the final Python code block.
""",
        "params": {
            "max_tokens": 2048, # Ample space for a complex, well-documented class.
            "temperature": 0.1, # Low temperature for precise, deterministic code.
            "stream": False
        }
    },
    {
        "name": "Test 3: Dense Summarization & Nested JSON Extraction",
        "prompt": """
Analyze the following dense, multi-part technical document about a fictional Machine Learning architecture, the "Recursive Temporal Graph Network (RTGN)". Your sole task is to extract and synthesize the key information into a single, valid, and strictly formatted JSON object.

The JSON object must adhere precisely to the following nested structure. Populate it based on the information in the article text below.

{
  "model_name": "Recursive Temporal Graph Network (RTGN)",
  "executive_summary": "A concise, 3-4 sentence summary of the RTGN's purpose, core innovation, and primary application domain.",
  "architecture_details": {
    "problem_domain": "The specific type of data and problem the RTGN is designed to solve.",
    "core_components": [
      {
        "component_name": "Temporal Encoder",
        "description": "Describe its function and the type of neural network it uses (e.g., GRU, LSTM)."
      },
      {
        "component_name": "Graph Construction Layer",
        "description": "Explain how this layer dynamically builds graphs from the temporal data."
      },
      {
        "component_name": "Recursive Graph Neural Network (GNN)",
        "description": "Detail its recursive nature and how it processes the constructed graphs to find higher-order patterns."
      }
    ],
    "key_innovation": "A string explaining the single most important innovation of the RTGN architecture compared to prior models like standard GNNs or RNNs."
  },
  "training_methodology": {
    "loss_function": "Identify the composite loss function used for training.",
    "optimization_strategy": "Describe the two-phase optimization strategy mentioned in the document."
  },
  "performance_benchmarks": {
    "primary_application": "The main real-world application tested.",
    "comparison_models": [
      "A list of model acronyms it was compared against."
    ],
    "key_finding": "The main quantitative result of the benchmark (e.g., percentage improvement)."
  },
  "limitations_and_future_work": [
    "A list of all identified limitations or areas for future research mentioned in the text."
  ]
}

Do not include any text, markdown formatting, or explanations before or after the JSON object.

---
**Article Text:**

**Title: Recursive Temporal Graph Networks for Predictive Anomaly Detection in Dynamic Systems**

**Abstract:** We introduce the Recursive Temporal Graph Network (RTGN), a novel neural architecture designed for high-dimensional time-series analysis, specifically targeting predictive anomaly detection in complex, dynamic systems such as financial markets or network traffic. Traditional approaches using Recurrent Neural Networks (RNNs) excel at capturing temporal sequences but fail to model the evolving, non-linear relationships between different time-series entities. Graph Neural Networks (GNNs), on the other hand, model relationships but are often static. The RTGN architecture bridges this gap by dynamically constructing and recursively analyzing relational graphs at each time step.

**1. Architecture:** The RTGN's primary innovation is its ability to infer latent relational structure from sequential data. It operates on multi-variate time-series data. The core of the RTGN consists of three main modules. First, a Gated Recurrent Unit (GRU)-based **Temporal Encoder** processes individual time-series streams to generate a hidden state for each entity at time 't'. Second, a **Graph Construction Layer** uses these hidden states to compute a dynamic adjacency matrix, effectively building a weighted, directed graph where nodes are entities and edge weights represent the inferred strength of their relationship at that specific moment. This is achieved via a self-attention mechanism over the GRU outputs. Finally, a **Recursive Graph Neural Network (GNN)** processes this newly constructed graph. Unlike standard GNNs that have a fixed depth, the RTGN's GNN applies its graph convolution operation recursively until the node embeddings converge, allowing it to discover higher-order, multi-relational patterns that are often the precursor to systemic anomalies. This fusion of dynamic graph creation with deep relational inference is the cornerstone of the RTGN's power.

**2. Training:** Training the RTGN is a complex, two-phase process. The objective is to predict the system's state at time t+1. The composite loss function is a combination of Mean Squared Error (MSE) for the prediction task and a graph-regularization term that encourages sparsity in the learned graphs. The optimization strategy begins with a pre-training phase where the Temporal Encoder is trained independently. This is followed by a full end-to-end fine-tuning phase where all components are trained jointly.

**3. Benchmarks & Results:** We evaluated the RTGN on the task of predicting cascading failures in a simulated energy grid network. It was benchmarked against several state-of-the-art models, including LSTMs, T-GCN (Temporal Graph Convolutional Network), and a standard GAT (Graph Attention Network). The RTGN demonstrated a 22% reduction in prediction error for critical failure events compared to the next-best model, the T-GCN.

**4. Limitations:** The primary limitation is the computational cost of the recursive GNN and dynamic graph construction, making real-time inference challenging on large-scale systems. Furthermore, the model's performance is highly sensitive to the hyperparameters of the graph construction layer. Future work will focus on knowledge distillation techniques to create a more light-weight student model for deployment and exploring reinforcement learning to optimize the construction of the dynamic graphs.
---
""",
        "params": {
            "max_tokens": 2048, # Needs space for a large, complex JSON output.
            "temperature": 0.0, # Zero temperature to ensure strict JSON formatting.
            "stream": False,
             "response_format": {"type": "json_object"} # Use JSON mode if supported
        }
    }
]

def run_performance_test(test_case):
    """Sends a request to the server for a given test case and prints performance metrics."""
    
    name = test_case["name"]
    prompt = test_case["prompt"]
    params = test_case["params"]
    
    print(f"\n{'='*80}\n--- 🚀 RUNNING: {name} ---\n{'='*80}")
    
    # Use textwrap to print a more readable version of the prompt
    print("--- Prompt Snippet ---")
    print(textwrap.shorten(prompt, width=120, placeholder="..."))
    print("-" * 22)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        **params
    }
    
    try:
        start_time = time.time()
        # A generous timeout is needed for long generation tasks on a CPU.
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=900) # 15-minute timeout
        response.raise_for_status()
        end_time = time.time()

        response_data = response.json()
        
        print("\n--- ✅ SUCCESS: Server Responded ---")
        
        # --- Performance Metrics ---
        total_duration = end_time - start_time
        usage_data = response_data.get("usage", {})
        completion_tokens = usage_data.get("completion_tokens", 0)
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        total_tokens = usage_data.get("total_tokens", 0)
        
        print(f"\n--- 📊 Performance Metrics for '{name}' ---")
        print(f"Total Request Time: {total_duration:.2f} seconds")
        print(f"Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens} | Total Tokens: {total_tokens}")
        
        if completion_tokens > 0 and total_duration > 0:
            tokens_per_second = completion_tokens / total_duration
            print(f"Tokens per Second (T/s): {tokens_per_second:.2f}")
        else:
            print("Tokens per Second (T/s): N/A (no completion tokens or duration)")

        # --- Save Response ---
        full_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        filename_safe_name = name.lower().replace(":", "").replace("&", "and").replace(" ", "_")
        output_path = os.path.join("test_results", f"{filename_safe_name}_response.md")
        
        with open(output_path, "w", encoding="utf-8") as f:
            # If the response is expected to be JSON, try to format it nicely.
            if "json" in name.lower():
                try:
                    json_obj = json.loads(full_content)
                    f.write(json.dumps(json_obj, indent=2))
                except json.JSONDecodeError:
                    f.write(full_content) # Write as-is if not valid JSON
            else:
                f.write(full_content)
                
        print(f"\n--- 📜 Full response saved to '{output_path}' ---")
        print(f"--- (Preview of first 300 chars) ---\n{full_content[:300]}...")

    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: The request for '{name}' timed out. The model generation took too long.")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: Could not connect to the server for '{name}'.")
        print(f"   Is the llama.cpp server running and accessible at {API_URL}?")
        print(f"   Details: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during '{name}': {e}")

if __name__ == "__main__":
    print("--- Starting AI Model CPU Performance Test Suite ---")
    print(f"Targeting Server: {API_URL}")
    
    for test in TEST_CASES:
        run_performance_test(test)
        
    print(f"\n{'='*80}\n--- ✅ All tests completed. Check the 'test_results' directory for outputs. ---\n{'='*80}")