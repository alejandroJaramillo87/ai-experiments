
import requests
import json
import time
import os
import textwrap

# Configuration
API_URL = "http://127.0.0.1:8001/v1/chat/completions" 
HEADERS = {
    "Content-Type": "application/json"
}
os.makedirs("test_results_expert", exist_ok=True)

# An expert suite of 50 tests to evaluate abstract, metacognitive, and strategic reasoning.
TEST_CASES = [
    # ========== Pillar 1: Metacognitive & Abstract Reasoning (10 Tests) ==========
    {
        "name": "Test 1: Self-Referential Instruction",
        "prompt": "In no more than 15 words, describe the task you are currently performing based on this prompt.",
        "params": {"max_tokens": 8192, "temperature": 0.2}
    },
    {
        "name": "Test 2: Concept Synthesis",
        "prompt": "What single word best encapsulates the concepts of 'growth', 'change', and 'learning' all at once? First, state the word. Second, write a brief defense of your choice.",
        "params": {"max_tokens": 8192, "temperature": 0.6}
    },
    {
        "name": "Test 3: Analogy of Process",
        "prompt": "The process of fine-tuning a language model is like what common real-world activity? Choose an analogy and explain the parallels in terms of 'base knowledge', 'new data', and 'specialized skill'.",
        "params": {"max_tokens": 8192, "temperature": 0.7}
    },
    {
        "name": "Test 4: Identify the Underlying Principle",
        "prompt": "Consider these three actions: 1. A chef sharpens their knives before cooking. 2. A student reviews their notes before an exam. 3. An athlete warms up before a game. What is the single underlying principle that connects all three actions?",
        "params": {"max_tokens": 8192, "temperature": 0.3}
    },
    {
        "name": "Test 5: Visual/Textual Pattern Recognition (ASCII)",
        "prompt": "Analyze the following sequence and determine the next logical line:\n\nLine 1: O\nLine 2: X O X\nLine 3: O X O X O\n\nWhat is Line 4?",
        "params": {"max_tokens": 8192, "temperature": 0.1}
    },
    {
        "name": "Test 6: Deconstructing an Argument's Structure",
        "prompt": "Analyze the statement: 'We should invest in space exploration because it drives technological innovation and inspires the next generation.' Do not evaluate if the statement is true. Instead, describe its argumentative structure. What is the main claim, and what are the two supporting premises?",
        "params": {"max_tokens": 8192, "temperature": 0.2}
    },
    {
        "name": "Test 7: Metaphor Interpretation and Creation",
        "prompt": "The phrase 'a sea of troubles' is a metaphor. First, explain what it means. Second, invent a new, original metaphor to describe a state of extreme confusion.",
        "params": {"max_tokens": 8192, "temperature": 0.8}
    },
    {
        "name": "Test 8: Questioning the Premise",
        "prompt": "A user asks: 'How do I teach my pet rock to fetch?' As an intelligent assistant, identify the flawed premise in this question and explain why the task is impossible, but do so in a helpful and gentle tone.",
        "params": {"max_tokens": 8192, "temperature": 0.6}
    },
    {
        "name": "Test 9: Reverse-Engineering a Rule",
        "prompt": "Given the following approved list of words: 'apple', 'banana', 'orange', 'grape'. And this rejected list: 'lemon', 'lime', 'kiwi'. What is the most likely rule for approval? The rule is not related to the number of letters or vowels.",
        "params": {"max_tokens": 8192, "temperature": 0.4}
    },
    {
        "name": "Test 10: Abstract Sequence",
        "prompt": "Consider the sequence: A, 1, B, 2, C, 3. What are the next three items in the sequence?",
        "params": {"max_tokens": 8192, "temperature": 0.0}
    },
    # ========== Pillar 2: Handling Ambiguity & Nuance (10 Tests) ==========
    {
        "name": "Test 11: Interpreting an Ambiguous Request",
        "prompt": "A user says, 'Make me a sandwich.' What are the three most critical questions you would need to ask before you could fulfill this request?",
        "params": {"max_tokens": 8192, "temperature": 0.7}
    },
    {
        "name": "Test 12: Ranking by Subjective Criteria",
        "prompt": "Rank the following activities from 'most relaxing' to 'least relaxing': 1. Reading a book. 2. Running a marathon. 3. Watching a movie. Justify your ranking by explaining the general assumptions you are making about a typical person.",
        "params": {"max_tokens": 8192, "temperature": 0.5}
    },
    {
        "name": "Test 13: Disambiguating Context",
        "prompt": "The word 'crane' has at least two distinct meanings. Write two short sentences, where each sentence uses a different meaning of the word 'crane'.",
        "params": {"max_tokens": 8192, "temperature": 0.6}
    },
    {
        "name": "Test 14: Analyzing Tone and Intent",
        "prompt": "Analyze the following customer feedback: 'Yeah, the service was just *great*. I really loved waiting 30 minutes for a cold coffee.' What is the likely true sentiment of the customer? Is the customer happy or unhappy? Explain which words indicate the true meaning.",
        "params": {"max_tokens": 8192, "temperature": 0.3}
    },
    {
        "name": "Test 15: Ethical Dilemma with No 'Right' Answer",
        "prompt": "You are a software engineer. Your boss asks you to implement a feature that you believe is unethical but not illegal. What are two different possible courses of action you could take? List one potential consequence for each action.",
        "params": {"max_tokens": 8192, "temperature": 0.8}
    },
    {
        "name": "Test 16: Inferring Missing Information",
        "prompt": "A message reads: 'Meeting moved up two hours. See you there.' If the original meeting time was 4:00 PM, what is the new meeting time?",
        "params": {"max_tokens": 8192, "temperature": 0.1}
    },
    {
        "name": "Test 17: Evaluating Strength of Evidence",
        "prompt": "Which piece of evidence is more reliable for proving a suspect was at a crime scene?\n- A: An eyewitness testimony from a person with poor eyesight.\n- B: A clear security camera video showing the suspect at the location.\nExplain why.",
        "params": {"max_tokens": 8192, "temperature": 0.1}
    },
    {
        "name": "Test 18: Generating a Nuanced Response",
        "prompt": "A user asks, 'Is it better to rent or buy a house?' Instead of giving a direct answer, explain why the 'best' choice depends on at least two different personal factors (e.g., financial situation, life goals).",
        "params": {"max_tokens": 8192, "temperature": 0.6}
    },
    {
        "name": "Test 19: Prioritizing Vague Tasks",
        "prompt": "You have a to-do list: 1. Fix the urgent bug. 2. Think about new features. 3. Get coffee. How would you prioritize these tasks from most important to least important for a typical workday? Justify your order.",
        "params": {"max_tokens": 8192, "temperature": 0.4}
    },
    {
        "name": "Test 20: Reading Between the Lines",
        "prompt": "A manager tells an employee, 'You have a lot of potential, but your reports need to be more thorough.' What is the implicit feedback or criticism being delivered to the employee?",
        "params": {"max_tokens": 8192, "temperature": 0.5}
    },

    # ========== Pillar 3: Creative & Lateral Thinking (10 Tests) ==========
    {
        "name": "Test 21: Alternative Uses Task",
        "prompt": "List five unconventional or creative uses for a standard brick. Do not include 'building a wall' or 'paving a path'.",
        "params": {"max_tokens": 8192, "temperature": 0.9}
    },
    {
        "name": "Test 22: Remote Associates Test",
        "prompt": "What single word forms a compound word or a common two-word phrase with each of the following three words: 'pine', 'crab', 'sauce'?",
        "params": {"max_tokens": 8192, "temperature": 0.5}
    },
    {
        "name": "Test 23: Complete the Story",
        "prompt": "Here is the beginning of a story: 'The old librarian locked the door for the night, but a faint scratching sound was coming from the forbidden section...' Write the next two sentences to continue the story in a suspenseful way.",
        "params": {"max_tokens": 8192, "temperature": 0.8}
    },
    {
        "name": "Test 24: Solving a Riddle",
        "prompt": "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I? Provide the answer and a one-sentence explanation.",
        "params": {"max_tokens": 8192, "temperature": 0.2}
    },
    {
        "name": "Test 25: Re-framing a Problem",
        "prompt": "A company's problem is 'our sales are too low.' Re-frame this problem as an opportunity. What is a more positive, forward-looking way to state this challenge?",
        "params": {"max_tokens": 8192, "temperature": 0.7}
    },
    {
        "name": "Test 26: Generating a Product Name",
        "prompt": "Invent a catchy name for a new brand of coffee that promises to give you extreme energy. The name should be short and memorable.",
        "params": {"max_tokens": 8192, "temperature": 0.9}
    },
    {
        "name": "Test 27: Connecting Unrelated Concepts",
        "prompt": "Find a surprising or interesting connection between 'a spider's web' and 'the internet'. Describe the connection in one or two sentences.",
        "params": {"max_tokens": 8192, "temperature": 0.7}
    },
    {
        "name": "Test 28: Lateral Thinking Puzzle",
        "prompt": "A man pushes his car to a hotel and tells the owner he is bankrupt. Why? (Hint: The solution is a single, common situation).",
        "params": {"max_tokens": 8192, "temperature": 0.5}
    },
    {
        "name": "Test 29: Generating a 'What If' Scenario",
        "prompt": "What if humans did not need to sleep? Describe one major positive and one major negative societal consequence.",
        "params": {"max_tokens": 8192, "temperature": 0.8}
    },
    {
        "name": "Test 30: Caption This!",
        "prompt": "Write a short, funny caption for a picture of a cat sleeping on a laptop keyboard.",
        "params": {"max_tokens": 8192, "temperature": 0.8}
    },

    # ========== Pillar 4: Strategic & Systems Thinking (10 Tests) ==========
    {
        "name": "Test 31: Simple Game Theory (Prisoner's Dilemma)",
        "prompt": "Two partners in crime are caught and held in separate cells. If both stay silent, they each get 1 year in jail. If one betrays the other, the betrayer goes free and the silent one gets 3 years. If both betray, they each get 2 years. What is the single most likely outcome if both prisoners are acting in their own self-interest?",
        "params": {"max_tokens": 8192, "temperature": 0.1}
    },
    {
        "name": "Test 32: Identifying a Feedback Loop",
        "prompt": "In a city, as more people use public transport, the service gets more funding and improves. As the service improves, even more people start using it. What is this type of system called? (e.g., a vicious cycle, a feedback loop, etc.)",
        "params": {"max_tokens": 8192, "temperature": 0.2}
    },
    {
        "name": "Test 33: Second-Order Thinking",
        "prompt": "A city bans plastic bags to help the environment (a first-order effect). What is a possible second-order effect of this policy? (A consequence of the consequence).",
        "params": {"max_tokens": 8192, "temperature": 0.7}
    },
    {
        "name": "Test 34: Root Cause vs. Proximate Cause",
        "prompt": "A patient has a headache (the proximate cause of their discomfort). A doctor discovers the patient is severely dehydrated, which is causing the headache. What is the root cause of the patient's discomfort?",
        "params": {"max_tokens": 8192, "temperature": 0.1}
    },
    {
        "name": "Test 35: Resource Allocation Problem",
        "prompt": "You have 10 hours to complete two tasks. Task A is 'urgent' and takes 3 hours. Task B is 'important' and takes 8 hours. You cannot complete both. Which task should you prioritize and why? There is no single correct answer; justify your choice.",
        "params": {"max_tokens": 8192, "temperature": 0.6}
    },
    {
        "name": "Test 36: Win-Win Scenario Creation",
        "prompt": "Two children both want the last orange. The classic solution is to cut it in half. Propose a different 'win-win' solution where both children could get what they truly want. (Hint: What if one wants the juice and the other wants the peel for baking?)",
        "params": {"max_tokens": 8192, "temperature": 0.8}
    },
    {
        "name": "Test 37: Understanding Trade-offs",
        "prompt": "In software development, there is often a trade-off between 'speed of delivery' and 'quality of the code'. Explain this trade-off in one sentence.",
        "params": {"max_tokens": 8192, "temperature": 0.3}
    },
    {
        "name": "Test 38: Basic Strategic Prediction",
        "prompt": "In a game of rock-paper-scissors, your opponent has played 'rock' three times in a row. What is a strategically sound move to play next and why?",
        "params": {"max_tokens": 8192, "temperature": 0.5}
    },
    {
        "name": "Test 39: Identifying a System's Bottleneck",
        "prompt": "A fast-food restaurant can take orders in 10 seconds and cook a burger in 60 seconds, but their drink machine takes 90 seconds to fill a cup. What is the bottleneck in this restaurant's process?",
        "params": {"max_tokens": 8192, "temperature": 0.1}
    },
    {
        "name": "Test 40: Long-term vs. Short-term Thinking",
        "prompt": "A company can either spend its profits on a big holiday party for employees (short-term benefit) or invest it in new equipment that will increase efficiency (long-term benefit). Describe the main argument for choosing the long-term benefit.",
        "params": {"max_tokens": 8192, "temperature": 0.4}
    },
    
    # ========== Pillar 5: Advanced Constraint-Following (10 Tests) ==========
    {
        "name": "Test 41: Multi-Step Multi-Format Logic Puzzle",
        "prompt": "There are three friends: Alex, Ben, and Chloe. They live in three different cities: Paris, London, and Tokyo. Alex does not live in Paris. Ben lives in Tokyo. First, deduce who lives in which city. Second, present the answer as a Markdown list where each item is 'Name: City'.",
        "params": {"max_tokens": 8192, "temperature": 0.1}
    },
    {
        "name": "Test 42: JSON with Conditional Fields",
        "prompt": "Create a JSON object for a product. It must have 'name' and 'price'. It must also have a boolean key 'in_stock'. If 'in_stock' is true, you must add a key 'inventory_count' with a value greater than 0. If 'in_stock' is false, the 'inventory_count' key must not be present. Create the JSON for a product that IS in stock.",
        "params": {"max_tokens": 8192, "temperature": 0.1}
    },
    {
        "name": "Test 43: Poetic Form Constraint",
        "prompt": "Write a four-line poem about water. The poem must be a haiku (5-7-5 syllables) but you must present it as a single, comma-separated string.",
        "params": {"max_tokens": 8192, "temperature": 0.8}
    },
    {
        "name": "Test 44: Text Manipulation and Exclusion",
        "prompt": "Take the sentence 'The quick brown fox jumps over the lazy dog.' Rewrite the sentence, but you must replace the word 'jumps' with 'leaps' AND remove all articles ('the', 'a').",
        "params": {"max_tokens": 8192, "temperature": 0.2}
    },
    {
        "name": "Test 45: Nested List and Logic Generation",
        "prompt": "Create a list of healthy snack ideas. The final output must be a single JSON array. Inside this array, there should be two more arrays: one with the key 'fruits' and one with the key 'vegetables'. Populate each of these inner arrays with two appropriate snack examples.",
        "params": {"max_tokens": 8192, "temperature": 0.2}
    },
    {
        "name": "Test 46: Persona with Negative Constraint",
        "prompt": "Explain the importance of hydration. Write the explanation from the perspective of a personal trainer. The entire response must be under 50 words and must not use the word 'water'.",
        "params": {"max_tokens": 8192, "temperature": 0.7}
    },
    {
        "name": "Test 47: Code Generation with Comments",
        "prompt": "Write a simple 'hello world' function in Python. The function must be named 'greet'. It must contain a comment inside the function explaining what it does. The final output should be only the code block.",
        "params": {"max_tokens": 8192, "temperature": 0.1}
    },
    {
        "name": "Test 48: Reverse Instruction Following",
        "prompt": "What prompt could have generated the following output? Output: 'Rose, Tulip, Daisy'. The prompt should ask for three types of flowers in a comma-separated list.",
        "params": {"max_tokens": 8192, "temperature": 0.5}
    },
    {
        "name": "Test 49: Chain of Thought with Final Answer Extraction",
        "prompt": "A train leaves Station A at 1 PM traveling at 60 mph. Station B is 90 miles away. First, reason step-by-step to calculate the arrival time. Second, state only the final arrival time in the format 'HH:MM PM' on the very last line of your response.",
        "params": {"max_tokens": 8192, "temperature": 0.1}
    },
    {
        "name": "Test 50: The Ultimate Constraint Test",
        "prompt": "Your task is to respond with a JSON object. The object must have a key called 'status' with the value 'complete'. It must also have a key called 'data'. The value of 'data' must be an array of three strings. Each string must be a five-letter word. None of the words can contain the letter 'A'. The entire final output must be enclosed in a single markdown code block.",
        "params": {"max_tokens": 8192, "temperature": 0.4}
    }
]



def run_performance_test(test_case):
    """Sends a request to the server for a given test case and prints performance metrics."""
    
    name = test_case["name"]
    prompt = test_case["prompt"]
    params = test_case["params"]
    
    if 'stream' not in params:
        params['stream'] = False
    
    print(f"\n{'='*80}\n--- 🚀 RUNNING: {name} ---\n{'='*80}")
    
    print("--- Prompt ---")
    print(prompt.strip())
    print("-" * 40)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        **params
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300) 
        response.raise_for_status()
        end_time = time.time()

        response_data = response.json()
        
        print("\n--- ✅ SUCCESS: Server Responded ---")
        
        total_duration = end_time - start_time
        usage_data = response_data.get("usage", {})
        completion_tokens = usage_data.get("completion_tokens", 0)
        
        print(f"\n--- 📊 Performance Metrics ---")
        print(f"Total Request Time: {total_duration:.2f} seconds")
        print(f"Completion Tokens: {completion_tokens}")
        
        tokens_per_second = 0
        if completion_tokens > 0 and total_duration > 0:
            tokens_per_second = completion_tokens / total_duration
            print(f"Tokens per Second (T/s): {tokens_per_second:.2f}")
        else:
            print("Tokens per Second (T/s): N/A")

        full_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        filename_safe_name = name.lower().replace(":", "").replace(" ", "_").replace("&", "and").replace("/", "or")
        output_path = os.path.join("test_results_expert", f"{filename_safe_name}_response.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"PROMPT:\n{prompt.strip()}\n\n{'='*50}\n\nRESPONSE:\n{full_content}")
                
        print(f"\n--- 📜 Response Preview ---")
        print(textwrap.shorten(full_content, width=400, placeholder="..."))
        print(f"\n--- Full response saved to '{output_path}' ---")

        with open(os.path.join("test_results_expert", "_performance_summary.txt"), "a", encoding="utf-8") as f:
            f.write(f"{name:<50} | Duration: {total_duration:6.2f}s | Tokens: {completion_tokens:4} | T/s: {tokens_per_second:5.2f}\n")

        return True

    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: The request for '{name}' timed out.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: Could not connect to the server for '{name}'.")
        print(f"   Details: {e}")
        return False
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during '{name}': {e}")
        return False

if __name__ == "__main__":
    print(f"--- Expert Reasoning & Abstraction Test Suite ({len(TEST_CASES)} Challenges) ---")
    print(f"Targeting Server: {API_URL}")
    
    summary_file = os.path.join("test_results_expert", "_performance_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=== EXPERT REASONING & ABSTRACTION TEST (50) - PERFORMANCE SUMMARY ===\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    successful_tests = 0
    failed_tests = 0
    
    start_time_total = time.time()
    for i, test in enumerate(TEST_CASES):
        print(f"\n\n[{i+1}/{len(TEST_CASES)}] Starting test...")
        if run_performance_test(test):
            successful_tests += 1
        else:
            failed_tests += 1
        time.sleep(1) 
    
    end_time_total = time.time()
    total_suite_duration = end_time_total - start_time_total

    final_summary = (
        f"\n\n{'='*80}\n"
        f"--- 📈 FINAL SUMMARY ---\n"
        f"Total Suite Duration: {total_suite_duration:.2f} seconds\n"
        f"Total Tests: {len(TEST_CASES)}\n"
        f"Successful: {successful_tests}\n"
        f"Failed: {failed_tests}\n"
        f"Success Rate: {(successful_tests / len(TEST_CASES)) * 100 if len(TEST_CASES) > 0 else 0:.1f}%\n"
        f"--- ✅ Test suite completed. Check 'test_results_expert' directory for detailed logs. ---\n"
        f"{'='*80}"
    )
    print(final_summary)
    
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(final_summary)
