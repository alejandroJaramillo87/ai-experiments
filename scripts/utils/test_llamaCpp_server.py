import requests
import json
import time

# 1. The URL points to your llama.cpp server's OpenAI-compatible endpoint.
API_URL = "http://127.0.0.1:8001/v1/chat/completions"

# 2. Standard header for JSON content.
headers = {
    "Content-Type": "application/json"
}

# 3. This long and structured prompt is designed to test the model's ability
#    to handle a large context and generate a lengthy, coherent response.
#    Using a triple-quoted string makes it easy to format.
long_prompt = """
Generate a comprehensive preliminary xenological report, designated File #X-77-B41, for the United Nations Extrasolar Council. The subject of the report is the recently discovered exoplanet Kepler-186f.

While you should ground your report in the known scientific data about Kepler-186f (e.g., its location in the habitable zone of a red dwarf), you must extrapolate and invent a highly detailed, complex, and plausible ecosystem. The tone should be formal, scientific, and rigorously detailed.

The report must follow this exact structure, with each section being thoroughly elaborated upon. Do not summarize; expand on every point with detailed descriptions, speculative analysis, and supporting rationale.

Report File #X-77-B41: Preliminary Xenological Survey of Kepler-186f

* 1.0 Executive Summary: A one-paragraph overview of the planet, its significance, and the key findings of this preliminary report, including the primary recommendation.

* 2.0 Planetary & Stellar System Data:
    * 2.1 Detailed description of the Kepler-186 star (red dwarf), including its energy output, solar flare activity, and spectral class.
    * 2.2 Kepler-186f's orbital characteristics: orbital period, semi-major axis, and the implications of its tidal-locking.
    * 2.3 Analysis of the planet's axial tilt and its effect on the terminator zone—the "ring of life" between the star-facing and dark sides.
    * 2.4 Geological composition: detailed analysis of the planet's suspected silicate and iron core, evidence of past and present tectonic activity, and significant mineral deposits detected via spectroscopy.

* 3.0 Atmospheric Composition & Climatology:
    * 3.1 Breakdown of atmospheric gases (Nitrogen, Oxygen, Argon, and significant trace gases, including a unique, unnamed noble gas).
    * 3.2 Cloud formations, weather patterns, and the perpetual storm systems on the star-facing side versus the calm, frozen conditions on the dark side.
    * 3.3 A deep analysis of the climatology within the terminator zone, detailing the gradient of temperatures, wind patterns, and precipitation.

* 4.0 Biosphere Analysis: Flora:
    * 4.1 The 'Crimson Weave': Describe the dominant photosynthesizing organism, a sprawling, interconnected network of vine-like flora that uses bacteriochlorophylls to absorb the red dwarf's light, giving the landscape its characteristic deep red and purple hues. Detail its structure, reproductive cycle, and role as the ecosystem's foundation.
    * 4.2 Bioluminescent Fungi: Detail the extensive fungal networks that thrive in the lower-light regions of the terminator zone. Describe their symbiotic relationship with the Crimson Weave and their use of bioluminescence for spore dispersal, creating moving constellations of light on the forest floor.
    * 4.3 Lithophytic Flora: Describe specialized, rock-like plants that have adapted to the high-radiation environment near the edge of the star-facing side, including their crystalline structures and slow metabolism.

* 5.0 Biosphere Analysis: Fauna:
    * 5.1 Herbivores - The 'Glyders': Describe the primary herbivores, six-limbed, low-gravity creatures that "glyde" through the dense Crimson Weave. Detail their anatomy, social structures, and feeding habits.
    * 5.2 Apex Predator - The 'Stalker of Twilight': Describe the primary predator that hunts within the terminator zone. Detail its hunting strategy, which relies on near-perfect camouflage and exploiting the sharp light gradients. Describe its sensory organs, adapted for both bright and low-light conditions.
    * 5.3 Subterranean Life: Speculate on the evidence of a complex ecosystem in the vast limestone cave networks beneath the surface, protected from solar radiation.

* 6.0 Technosignature Assessment & Anomaly Report:
    * 6.1 Radio Signal Analysis: Report on the complete lack of any structured, artificial radio signals.
    * 6.2 The 'Geometric Scars': Detail the discovery of continent-spanning geometric patterns on the surface, primarily vast hexagons and spirals etched into the landscape. They appear ancient and partially overgrown by the Crimson Weave. Analyze at least three competing hypotheses for their origin:
        * Hypothesis A: A natural, but currently unexplained, geological or crystalline formation process.
        * Hypothesis B: The remnants of a planet-wide biological organism, now extinct.
        * Hypothesis C: Evidence of a previous, non-extant technological civilization (i.e., archeological ruins).
    * 6.3 Rationale for Prioritizing Hypothesis C: Provide a detailed argument for why the precision, scale, and non-fractal nature of the patterns strongly suggest an artificial origin over natural phenomena.

* 7.0 Recommendations & Conclusion:
    * 7.1 Primary Recommendation: Formally recommend the immediate dispatch of a robotic exploration mission (the 'Magellan 2' probe) to conduct on-site analysis.
    * 7.2 Risk Assessment: Outline potential risks of a mission, including biological contamination (both ways) and unknown threats related to the 'Geometric Scars'.
    * 7.3 Concluding Remarks: Summarize why Kepler-186f represents the most significant exobiological discovery in human history.
"""

# 4. The payload for the API request.
#    The `model` parameter is removed because the server is started with a specific model.
#    `max_tokens` is set high to allow for a full response.
payload = {
    "messages": [
        {
            "role": "user",
            "content": long_prompt
        }
    ],
    "max_tokens": 8200,
    "temperature": 0.3, # A slightly higher temperature for more creative, detailed text.
    "stream": False
}

def test_cpu_long_inference():
    """Sends a long-form request to the CPU server and prints the response and timing."""
    print("--- 🧪 Sending LONG-FORM request to CPU Llama.cpp Server ---")
    print(f"URL: {API_URL}")
    
    try:
        start_time = time.time()
        # A very generous timeout is needed for this long generation task on a CPU.
        response = requests.post(API_URL, headers=headers, json=payload, timeout=900) # 15 minutes
        response.raise_for_status()
        end_time = time.time()

        response_data = response.json()
        
        print("\n--- ✅ SUCCESS: Server Responded ---")
        
        # Print metadata from the response
        total_duration = end_time - start_time
        completion_tokens = response_data.get("usage", {}).get("completion_tokens", 0)
        prompt_tokens = response_data.get("usage", {}).get("prompt_tokens", 0)
        
        print(f"\n--- 📊 Performance Metrics ---")
        print(f"Total Request Time: {total_duration:.2f} seconds")
        print(f"Prompt Tokens: {prompt_tokens}")
        print(f"Completion Tokens: {completion_tokens}")
        if completion_tokens > 0 and total_duration > 0:
            tokens_per_second = completion_tokens / total_duration
            print(f"Tokens per Second: {tokens_per_second:.2f}")

        print("\n--- 📜 Server Response Content (first 500 chars) ---")
        full_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(full_content[:500] + "...")
        
        # Save the full response to a file for review
        with open("long_form_response.md", "w", encoding="utf-8") as f:
            f.write(full_content)
        print("\nFull response saved to 'long_form_response.md'")

    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: The request timed out. The model generation took too long.")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: Could not connect to the server.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_cpu_long_inference()