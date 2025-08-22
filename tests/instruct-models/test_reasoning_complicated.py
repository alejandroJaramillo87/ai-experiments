import requests
import json
import time
import os
import textwrap

# Configuration
API_URL = "http://127.0.0.1:8004/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json"
}
os.makedirs("test_results", exist_ok=True)

TEST_CASES = [
    # ========== PURE LOGIC AND DEDUCTION (1-15) ==========
    {
        "name": "Test 1: Multi-Layer Deductive Chain",
        "prompt": """
Consider the following complex scenario with multiple interconnected rules:

In the nation of Logica, there are five provinces: Northland, Southland, Eastland, Westland, and Centraland. Each province has exactly one governor, and these governors follow strict protocols:

1. If Northland's governor attends a meeting, then either Southland's or Eastland's governor must also attend, but not both.
2. Westland's governor attends a meeting if and only if at least three other governors attend.
3. Centraland's governor never attends the same meetings as Northland's governor.
4. If Southland's governor attends, then Centraland's governor must also attend.
5. Eastland's governor attends every meeting where exactly two other governors are present.
6. There must be at least two governors at every official meeting.
7. If Westland's governor doesn't attend, then either Northland's or Southland's governor must attend, but not both.

Question: For next week's critical infrastructure meeting, Eastland's governor has already confirmed attendance. Based solely on these rules, determine:
a) Which governors MUST attend
b) Which governors CANNOT attend  
c) Which governors MIGHT attend (attendance is possible but not required)
d) Is there more than one valid configuration? If so, enumerate all possibilities.

Provide detailed logical reasoning for each conclusion, showing how you applied each rule and resolved any apparent contradictions. Explain why certain combinations are impossible and how the constraints interact to limit the possible configurations.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 2: Temporal Logic Puzzle",
        "prompt": """
Five witnesses gave testimonies about a series of events that occurred in a mansion during a thunderstorm. Each witness is either always truthful, always lies, or alternates between truth and lies with each statement they make. Analyze their testimonies:

Witness A states:
1. "The power went out before the vase was broken."
2. "Witness B is a constant liar."
3. "The butler was in the kitchen when the lights went out."
4. "I saw Witness C in the library after the crash."

Witness B states:
1. "Witness A alternates between truth and lies."
2. "The vase was broken while the lights were still on."
3. "Witness D was with me when we heard the crash."
4. "The butler left the kitchen before the power outage."

Witness C states:
1. "I am always truthful."
2. "Witness B's second statement is false."
3. "The crash happened exactly at midnight."
4. "Witness E is an alternator."

Witness D states:
1. "Witness C's first statement is a lie."
2. "I was alone when the crash occurred."
3. "The power went out at 11:58 PM."
4. "Witness A is always truthful."

Witness E states:
1. "At least two witnesses here are constant liars."
2. "The crash happened after midnight."
3. "Witness D's third statement is true."
4. "The butler was in three different rooms that night."

Determine:
1. The truth-telling pattern of each witness (always true, always false, or alternating)
2. The actual sequence of events
3. The location of each person at critical moments
4. Which statements are true and which are false

Explain your reasoning process, including how you resolved contradictions and used the constraints of each witness type to narrow down possibilities. Show why your solution is the only consistent one.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 3: Epistemic Modal Logic",
        "prompt": """
In a research facility, five scientists (Alice, Bob, Carol, David, and Eve) are working on different aspects of a classified project. The security protocols create interesting knowledge dynamics:

Initial Setup:
- Alice knows the project's true purpose
- Bob knows what Alice knows, but doesn't know that he knows it
- Carol knows that Bob knows something important, but doesn't know what
- David knows that Carol doesn't know the full picture
- Eve knows what everyone else knows and doesn't know

New Information Arrives:
A memo is circulated that states: "Anyone who knows the project's true purpose must know that they know it." However, the memo itself doesn't reveal the project's purpose.

After reading the memo:
1. Bob realizes something about his own knowledge
2. Carol deduces something from Bob's reaction
3. David updates his beliefs about what Carol knows
4. Eve observes everyone's reactions

Additional Constraints:
- If someone knows X and knows that they know X, they act differently in meetings
- Anyone who acts differently is noticed by Eve
- Carol is very perceptive and can deduce when someone has had a realization
- David always assumes others know less than they actually do

Questions:
1. After the memo, what does each scientist know about the project?
2. What does each scientist know about what the others know?
3. Who will act differently in the next meeting and why?
4. Can Eve deduce the project's true purpose from observing reactions? Explain.
5. Is there a state where everyone knows the truth but doesn't know that everyone knows? 

Detail the epistemic states at each stage and explain how knowledge propagates through the group. Address the distinction between knowing something and knowing that you know it.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 4: Counterfactual Reasoning Chain",
        "prompt": """
Analyze this complex scenario involving multiple counterfactual conditions:

In the country of Hypothetica, five major decisions were made last year:
- Decision A: Implement a new tax system
- Decision B: Build a national railway
- Decision C: Establish universal healthcare
- Decision D: Increase military spending
- Decision E: Invest in renewable energy

The actual outcomes were:
- The economy grew by 3%
- Unemployment fell to 4%
- Public satisfaction increased to 65%
- The deficit increased by $50 billion
- Carbon emissions decreased by 10%

Consider these counterfactual relationships:
1. If Decision A hadn't been made, Decision C would have been impossible
2. If Decision B had been made differently (local instead of national), Decision E would have been unnecessary
3. Had Decision C not occurred, either Decision D or E would have been doubled in scope
4. If both A and B hadn't happened, the country would have had to choose exactly two from C, D, and E
5. Decision D was only possible because Decision A was implemented first
6. Had Decision E been twice as large, Decision B would have been impossible

Additional counterfactual claims by analysts:
- Analyst 1: "Without Decision A, economic growth would have been negative"
- Analyst 2: "If we had skipped Decision B, we could have afforded both double C and double E"
- Analyst 3: "Decision D was unnecessary; without it, all other outcomes would have been better"
- Analyst 4: "Had we done none of these decisions, we'd be better off"
- Analyst 5: "The only essential decisions were A and C"

Evaluate:
1. Which analyst's counterfactual claims are logically consistent with the given relationships?
2. What would have been the minimal set of decisions to achieve positive economic growth?
3. Is there a combination that would have been impossible given the constraints?
4. What can we definitively conclude about the necessity of each decision?
5. Which counterfactual scenarios lead to logical contradictions?

Explain your reasoning about causation, dependency, and the validity of counterfactual claims.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 5: Recursive Belief Paradox",
        "prompt": """
In the village of Metacognition, there's a peculiar tradition regarding beliefs about beliefs:

The Rules:
1. Every villager must have a belief about what the majority believes
2. A villager is "enlightened" if their belief about the majority belief is correct
3. The majority is defined as more than 50% of villagers
4. Each villager knows the rules and knows that everyone else knows them

The Situation:
There are 100 villagers. A question is posed: "Is the village well-governed?"

Initial beliefs:
- 40 villagers believe "Yes, the village is well-governed"
- 60 villagers believe "No, the village is not well-governed"

However, regarding what they think the majority believes:
- 70 villagers believe that the majority thinks "Yes"
- 30 villagers believe that the majority thinks "No"

The Paradox Emerges:
The village elder announces: "Those who are enlightened about the majority belief will have their actual belief count double in future decisions."

This creates recursive complexity:
1. The "enlightened" are those 30 who correctly believe the majority thinks "No"
2. But if their beliefs count double, does this change what the majority believes?
3. If the majority belief changes, who is actually enlightened?
4. If the enlightened set changes, how does this affect the weighting?

Additional Complications:
- Each villager can reason about this recursion
- They can update their beliefs about what the majority believes
- But updating might change their enlightenment status
- Some villagers realize this creates an infinite loop

Questions:
1. Is there a stable configuration where the system reaches equilibrium?
2. Can a villager guarantee their own enlightenment through strategic belief?
3. What happens if villagers coordinate their beliefs about beliefs?
4. Is the elder's rule logically implementable, or does it create an unresolvable paradox?
5. How many levels of "belief about belief about belief" are necessary to analyze this fully?

Explore the recursive nature of the problem and whether any resolution exists.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 6: Impossible Object Analysis",
        "prompt": """
Consider these five objects that supposedly exist in the Museum of Logical Impossibilities. Each object has properties that seem contradictory, but the museum claims one of them actually could exist under the right interpretation:

Object 1: The Truthful Mirror
- Shows what will happen tomorrow
- Can be used to prevent what it shows
- Always shows the truth
- Has been used successfully multiple times

Object 2: The Complete Map
- Contains a perfect representation of everything, including itself
- The representation of itself contains a representation of the representation
- Is finite in size
- Can be consulted for accurate information

Object 3: The Democratic Dictator's Crown
- Grants absolute power to the wearer
- Can only be worn by someone chosen by unanimous vote
- The vote must be free and uncoerced
- The power cannot be used to influence future votes
- The wearer must obey the will of the people

Object 4: The Forgetting Stone
- Makes you forget something specific when touched
- You choose what to forget while touching it
- Once forgotten, you can't remember what you chose to forget
- You always remember using the stone
- You can use it multiple times successfully

Object 5: The Probability Compass
- Points to what you most need to find
- What you "need" is determined by future consequences
- Following it changes those future consequences
- It's always accurate about the original future
- It updates continuously as you move

Analyze each object:
1. What logical contradictions does each object contain?
2. Are there any interpretations under which an object could exist?
3. Which paradoxes do these objects embody?
4. Could any of these objects exist in a consistent logical framework?
5. What would be the implications if such an object did exist?

Explain your reasoning about possibility, self-reference, and logical consistency. Identify which object (if any) could actually exist and why.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 7: Transitive Property Maze",
        "prompt": """
In a logic competition, participants must navigate relationships that follow modified transitive properties:

Relationship Types:
- "Defeats": If A defeats B and B defeats C, then A defeats C (standard transitivity)
- "Trusts": If A trusts B and B trusts C, then A might trust C (probabilistic transitivity - 70% chance)
- "Knows": If A knows B and B knows C, then A knows about C but doesn't necessarily know C
- "Owes": If A owes B and B owes C, then C can claim from A only if all three agree
- "Teaches": If A teaches B and B teaches C, then A cannot teach C (anti-transitivity)

The Network:
- Alice defeats Bob, Bob defeats Carol, Carol defeats David, David defeats Eve, Eve defeats Alice
- Bob trusts Carol, Carol trusts David, David trusts Eve, Eve trusts Alice, Alice trusts Bob
- Carol knows David, David knows Eve, Eve knows Alice, Alice knows Bob, Bob knows Carol
- David owes Eve, Eve owes Alice, Alice owes Bob, Bob owes Carol, Carol owes David
- Eve teaches Alice, Alice teaches Bob, Bob teaches Carol, Carol teaches David, David teaches Eve

Complications:
1. If someone defeats and trusts the same person, one relationship negates
2. Knowing about someone through transitivity creates a "weak knowledge" relationship
3. Circular debt can be resolved only if there's a trust path connecting all parties
4. Teaching relationships can be inherited: if you can't teach someone, neither can your students
5. If a defeat cycle exists, all victories within it are nullified

Questions:
1. After applying all rules and complications, who actually defeats whom?
2. What is the probability that Alice trusts David? Eve?
3. Can the circular debt be resolved? If so, what agreements are needed?
4. Who can teach whom after considering anti-transitivity and inheritance?
5. How many "weak knowledge" relationships exist?
6. Are there any logical inconsistencies in the network? If so, how would you resolve them?

Show your step-by-step analysis of how the relationships evolve and interact.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 8: Gödel's Island Variant",
        "prompt": """
On an island, there are 100 inhabitants who follow strange logical rules. Each inhabitant is either a Knight (always tells truth), a Knave (always lies), or a Joker (makes statements that are neither true nor false - they are paradoxical or undefined).

The island has a peculiar property: Any statement about the island's properties that creates a logical paradox becomes true by making the speaker a Joker.

You meet five inhabitants who make the following statements:

Person A: "At least 60 inhabitants are Knaves, and I am not one of them."

Person B: "Person A is a Joker if and only if this statement is false."

Person C: "The number of Knights equals the number of Knaves, and there are exactly 20 Jokers."

Person D: "If I am a Knight, then Person B is a Knave. If I am a Knave, then Person B is a Knight. If I am a Joker, then Person B is also a Joker."

Person E: "This statement is true if and only if I am not a Joker."

Additionally, the Island Oracle (who exists outside the Knight/Knave/Joker system) tells you: "The total number of true statements made by these five people equals the number of Jokers among them."

Determine:
1. What type is each person (Knight, Knave, or Joker)?
2. Which statements are true, false, or paradoxical?
3. How does the island's peculiar property affect the logical analysis?
4. Is the Oracle's statement consistent with your conclusions?
5. What can we deduce about the actual distribution of types on the island?
6. Are there multiple valid solutions, or is the answer unique?

Explain how self-reference and paradox resolution work in this system.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 9: The Metacognitive Tournament",
        "prompt": """
Five philosophers enter a tournament where scoring depends on predicting both outcomes and other players' predictions:

Round Structure:
- Each philosopher predicts who will win the round
- They also predict what each other philosopher will predict
- Points are awarded for correct predictions at both levels
- The winner is determined by who has the most correct predictions

The Philosophers' Strategies:
- Aristotle: Always predicts the most logical outcome based on past performance
- Berkeley: Predicts based on what he believes others perceive to be likely
- Confucius: Seeks harmony by predicting the outcome that creates the least conflict
- Descartes: Doubts everything and predicts the least expected outcome
- Epicurus: Predicts whatever leads to the most interesting paradox

Historical Performance:
- In previous tournaments, Aristotle won 40% of rounds
- Berkeley won 25% of rounds
- Confucius won 20% of rounds
- Descartes won 10% of rounds
- Epicurus won 5% of rounds

The Metacognitive Twist:
Each philosopher knows:
1. Everyone's historical performance
2. Everyone's strategy
3. That everyone knows everyone's strategy
4. That this knowledge affects predictions

For the final round:
- Aristotle predicts Aristotle will win (based on history)
- Berkeley predicts Confucius will win (believes others see this as harmonious)
- Confucius predicts Berkeley will win (least conflicting given the predictions)
- Descartes predicts Epicurus will win (least expected)
- Epicurus predicts Descartes will win (creates maximum paradox)

Questions:
1. What should each philosopher predict about others' predictions?
2. Given these second-level predictions, who actually wins the round?
3. Is there a stable Nash equilibrium in predictions?
4. How does knowing that everyone knows everyone's strategy affect the outcome?
5. Can Epicurus create a genuine paradox in this system?
6. What would happen if they could recursively update predictions?

Analyze the levels of reasoning and metacognition involved.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 10: The Impossible Crime",
        "prompt": """
A crime has been committed in a sealed room with five suspects. The evidence creates a logical puzzle where seemingly no one could be guilty:

The Setup:
- The victim was poisoned at exactly 3:00 PM
- The poison takes exactly 30 minutes to work
- The room was locked from inside at 2:00 PM
- No one entered or left between 2:00 PM and 4:00 PM
- The poison was in the victim's coffee

The Suspects and Their Alibis:
1. Anna: "I was video-calling my lawyer from 2:30 to 3:30. The recording proves it."
2. Brian: "I'm allergic to coffee and never went near the coffee station."
3. Claire: "I was unconscious from 2:45 to 3:15 - the medical records confirm it."
4. Daniel: "I prepared the coffee at 1:45, before the room was sealed."
5. Emma: "I was handcuffed to the radiator from 2:15 onwards as part of a magic trick gone wrong."

Additional Evidence:
- The coffee was fresh when consumed (prepared within 30 minutes)
- Security footage shows all five suspects in the room at 3:00 PM
- The poison was added to the coffee after it was prepared
- Everyone's alibi has been verified as technically true
- The victim drank the coffee voluntarily
- No automated systems were in the room

Complicating Factors:
- If Anna was on video, she couldn't have physically poisoned the coffee
- If Brian never touched coffee, he couldn't have poisoned it
- If Claire was unconscious, she couldn't have acted
- If Daniel prepared coffee at 1:45, it wouldn't be fresh at 3:00
- If Emma was handcuffed, she couldn't have reached the coffee

Yet someone must be guilty. Resolve this impossible crime:
1. Who is the murderer?
2. How did they commit the crime despite their alibi?
3. What logical loophole or assumption makes this possible?
4. Why do all the alibis appear true yet don't exclude guilt?
5. What is the precise timeline of events?

Explain how apparent impossibilities can coexist with necessary truths.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 11: The Prediction Market Paradox",
        "prompt": """
A prediction market exists where traders bet on future events, but with unusual rules that create logical complexities:

Market Rules:
1. Traders can bet on whether events will occur
2. They can also bet on what the market price will be
3. Market prices are publicly visible and update in real-time
4. If the market price for an event reaches 90%, the event becomes more likely to occur
5. If the market price drops below 10%, the event becomes less likely to occur

The Scenario:
Five expert traders are betting on: "Company X will be acquired within 30 days"

Trader Positions and Reasoning:
- Trader A: Believes acquisition probability is 60%, but will bet based on market manipulation potential
- Trader B: Has inside information that acquisition is 80% likely, but knows others might have better info
- Trader C: Trades purely on market momentum and reflexivity
- Trader D: Always bets against the consensus when price exceeds 70% or below 30%
- Trader E: Has the power to actually influence the acquisition decision based on market sentiment

Current situation:
- Market price starts at 50%
- Each trader knows the others' strategies
- Each trader has limited capital (can't dominate the market alone)
- The CEO of Company X watches the market and may change decisions based on it

The Reflexive Loop:
- If traders believe the price will rise, they buy, causing it to rise
- Rising prices make the acquisition more likely (Rule 4)
- Higher likelihood justifies higher prices
- But if everyone knows this, should they trade on fundamental value or expected market dynamics?

Questions:
1. What price should each trader rationally bet toward?
2. Can the market reach a stable equilibrium, or will it oscillate?
3. How does Trader E's influence create a self-fulfilling prophecy?
4. Is there a price where the market's prediction becomes necessarily true?
5. How do recursive expectations affect the rational strategy?
6. Can the market simultaneously be efficient and self-manipulating?

Analyze the feedback loops between prediction, action, and outcome.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 12: The Semantic Paradox Network",
        "prompt": """
In a philosophy department, five professors each proposed a thesis. Each thesis makes claims about the truth values of the other theses, creating an interconnected web of semantic dependencies:

Professor Alpha's Thesis: "Beta's thesis is true if and only if Gamma's thesis is false."

Professor Beta's Thesis: "If my thesis is true, then Delta's thesis is false. If my thesis is false, then Epsilon's thesis is true."

Professor Gamma's Thesis: "Exactly two of the five theses are true, and Alpha's is one of them."

Professor Delta's Thesis: "The number of true theses equals the number of false theses that reference true theses."

Professor Epsilon's Thesis: "This thesis is true if and only if an odd number of the other theses are self-referential."

Definitions:
- A thesis is "self-referential" if its truth value depends on itself
- A thesis "references" another if its truth depends on the other's truth value
- The theses must all have definite truth values (true or false, not undefined)

Additional Constraints:
1. The department requires logical consistency across all theses
2. No thesis can be true by virtue of circular reasoning alone
3. Each professor knows the content of all other theses
4. The truth values must be determinable through pure logic

Questions:
1. Which theses are self-referential?
2. What are the truth values of each thesis?
3. Are there multiple consistent solutions, or is the answer unique?
4. How do you break the circular dependencies to find a solution?
5. What happens if we remove the constraint that theses must have definite truth values?
6. Is Delta's thesis even logically coherent given its self-referential nature?

Demonstrate how to resolve interconnected semantic paradoxes and find consistent truth assignments.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 13: The Evolutionary Strategy Puzzle",
        "prompt": """
On an isolated island, five species have evolved unique survival strategies that create a complex logical ecosystem:

Species and Their Rules:
- Alphas: Thrive when exactly two other species have populations over 1000
- Betas: Population doubles when Alphas are below 500, halves when Alphas exceed 1500
- Gammas: Can only survive if Betas outnumber Deltas
- Deltas: Grow by consuming Gammas, shrink without them
- Epsilons: Population equals 3000 minus (Alpha population / 2)

Starting Populations:
- Alphas: 1000
- Betas: 1000
- Gammas: 1000
- Deltas: 1000
- Epsilons: 2500

Interaction Rules:
1. Populations update simultaneously each generation
2. If a species drops to 0, it's extinct and cannot recover
3. Populations cannot be negative or fractional
4. Each species follows its rule based on the previous generation's numbers
5. The ecosystem reaches equilibrium when no populations change

Additional Complexities:
- If Alphas thrive, they suppress Betas in the next generation
- If Deltas consume all Gammas, they experience a population crash
- Epsilons can sacrifice 500 population to prevent any other species from going extinct
- There's a "cascade point" where the system becomes chaotic
- Some configurations lead to oscillating populations

Questions:
1. What happens to each population after 5 generations?
2. Is there a stable equilibrium? If so, what are the equilibrium populations?
3. Which species are at risk of extinction?
4. Can the Epsilons' sacrifice ability save the ecosystem?
5. What initial conditions would lead to all species surviving?
6. Is there a configuration where the logical rules become contradictory?

Analyze how interdependent logical rules create emergent behaviors and identify stable states.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 14: The Information Cascade Paradox",
        "prompt": """
In a small town, five influential citizens must decide whether to support a new policy. Each has private information but also observes others' choices, creating an information cascade with logical complications:

The Citizens and Their Information:
- Citizen A: Has weak evidence against the policy (40% confidence it's bad)
- Citizen B: Has strong evidence for the policy (70% confidence it's good)
- Citizen C: Has no private information, relies entirely on others
- Citizen D: Has contradictory information (evidence both for and against)
- Citizen E: Has meta-information about the quality of others' information

Decision Rules:
1. Citizens declare in order: A, then B, then C, then D, then E
2. Each can see all previous declarations but not the reasoning
3. Citizens are rational and want to make the correct decision
4. They update beliefs based on Bayesian reasoning
5. If uncertain (50/50), they follow the majority of previous decisions

The Cascade Complication:
- A knows that B has better information sources generally
- B knows that A tends to decide correctly despite weak information
- C knows that both A and B sometimes vote strategically
- D knows something that would change everyone's mind but can't communicate it
- E knows the exact confidence levels of A and B

Strategic Considerations:
- Citizens might vote against their information to influence later voters
- Knowing someone votes strategically changes how you interpret their vote
- The final outcome affects everyone equally
- Citizens care about both being right and the group making the right choice

The Meta-Paradox:
If everyone knows that everyone might vote strategically, how should they interpret votes? And if they can't interpret votes, should they vote sincerely or strategically?

Questions:
1. How should each citizen vote if they're purely rational?
2. What if each citizen knows the others are also rational?
3. Can D's contradictory information be logically incorporated?
4. How does E's meta-information affect the cascade?
5. Is there a Nash equilibrium in voting strategies?
6. What happens if citizens can recursively reason about each other's reasoning?

Explore how information cascades interact with strategic reasoning and recursive thinking.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 15: The Quantum Logic Room",
        "prompt": """
A thought experiment involving five observers in a room where classical logic doesn't fully apply, similar to quantum superposition but for logical states:

The Setup:
- Five observers: Alice, Bob, Carol, David, and Eve
- Each observer can be in a state of "knowing" or "not knowing" a secret
- The states can be in "superposition" - neither definitely knowing nor not knowing
- Observation by one person collapses another's state
- The secret is: "The majority of observers will eventually know the secret"

Initial States:
- Alice: Definitely knows the secret
- Bob: In superposition between knowing and not knowing
- Carol: Definitely doesn't know
- David: In superposition, entangled with Bob (same state when observed)
- Eve: In a state that's the opposite of the majority

Observation Rules:
1. When someone observes another, the observed person's state collapses
2. Observation is mutual - both parties' states may change
3. If the secret becomes false, all knowledge of it becomes uncertain
4. Superposition states count as 0.5 for determining majority
5. Eve's state automatically adjusts to maintain opposition to majority

The Logical Paradox:
- If majority knows → secret is true → Eve doesn't know → majority might not know
- If majority doesn't know → secret is false → knowledge becomes uncertain
- Observation changes the system, potentially changing the secret's truth

Sequence of Events:
1. Alice observes Bob
2. Carol observes David
3. Eve observes Alice
4. Bob observes Carol
5. David observes Eve

Questions:
1. What is each person's state after all observations?
2. Is the secret true or false at the end?
3. How do superposition states resolve into definite states?
4. Can the system reach a logically consistent end state?
5. What happens to Eve's state given her paradoxical definition?
6. Is there an observation sequence that creates an unresolvable paradox?

Analyze how quantum-like superposition of logical states creates new types of paradoxes and whether classical logical reasoning can resolve them.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },

    # ========== CAUSAL REASONING AND ANALYSIS (16-30) ==========
    {
        "name": "Test 16: Causal Chain Reversal",
        "prompt": """
Analyze this complex scenario where apparent causal chains might actually run in reverse:

Five events occurred in a laboratory, and scientists are debating their causal relationships:
- Event A: The temperature in the room dropped suddenly
- Event B: A chemical reaction stopped prematurely
- Event C: The emergency ventilation system activated
- Event D: A container's pressure increased rapidly
- Event E: The lights flickered and dimmed

Observed Correlations:
- A and B always occur within 2 seconds of each other
- C happens 5 seconds after B, without exception
- D occurs before A in 60% of cases, after A in 40%
- E happens randomly relative to other events

Scientists' Theories:
Scientist 1: "A causes B, B causes C, D causes A sometimes, E is independent"
Scientist 2: "B causes A and C, D is caused by a hidden variable that also affects A"
Scientist 3: "C causes B retroactively through quantum mechanics, B causes A"
Scientist 4: "All events are caused by a common hidden cause with different delays"
Scientist 5: "The causal chain is circular: A→B→C→D→A"

Additional Evidence:
- When A is artificially prevented, B still occurs 50% of the time
- When B is prevented, C never occurs but A still happens
- Preventing C doesn't affect B's occurrence
- D can occur without any other events
- E's occurrence slightly increases the probability of A

Experimental Constraints:
- Only one event can be artificially controlled at a time
- Some causal influences might have time delays
- Quantum retrocausation is theoretically possible in this system
- The system might have multiple stable states

Questions:
1. Which scientist's theory best fits the evidence?
2. What additional experiments would definitively establish causation?
3. Is reverse causation (future affecting past) necessary to explain the data?
4. How do you distinguish correlation from causation in this scenario?
5. What is the most likely true causal structure?
6. Could multiple theories be simultaneously correct?

Explain your reasoning about causation, correlation, and the possibility of retrocausal effects.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 17: The Blame Attribution Network",
        "prompt": """
A complex system failure occurred involving five components, and investigators must determine the root cause and attribute blame fairly:

The System:
- Component A: Data input validator
- Component B: Processing engine
- Component C: Safety checker
- Component D: Output formatter
- Component E: System monitor

The Failure Sequence:
1. At 10:00:00 - A received invalid data but marked it as valid
2. At 10:00:01 - B processed the data and produced errors
3. At 10:00:02 - C detected anomalies but didn't stop the process
4. At 10:00:03 - D formatted corrupted output
5. At 10:00:04 - E noticed irregularities but alert failed
6. At 10:00:05 - System crash with data loss

Component Interdependencies:
- A relies on E's monitoring to catch edge cases
- B assumes A has validated all input properly
- C's thresholds are calibrated based on B's normal output
- D can only format what it receives from B and C
- E's alerting depends on configurations set by C

Design Responsibilities:
- A was designed to catch 99% of invalid inputs
- B was designed to handle some invalid inputs gracefully
- C was supposed to be the final safety net
- D should preserve data integrity even with bad input
- E should have redundant alerting mechanisms

Mitigating Factors:
- A was operating with outdated validation rules
- B was running in a degraded mode due to maintenance
- C had its sensitivity reduced after false positives
- D was never tested with this type of corrupted data
- E's primary alert channel was undergoing maintenance

The Blame Questions:
1. Which component bears primary responsibility for the failure?
2. How should blame be distributed among the five components?
3. Is the system design itself at fault rather than any component?
4. How do you weigh design flaws versus operational failures?
5. Should components that failed earlier in the chain bear more blame?
6. How does the interdependency affect moral and causal responsibility?

Consider proximate versus ultimate causes, design versus implementation failures, and the philosophy of blame in complex systems.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 18: Emergent Causation Puzzle",
        "prompt": """
In a simulated society, five simple rules governing individual behavior lead to complex emergent phenomena that seem to violate causation:

Individual Rules:
1. Move toward the average position of your three nearest neighbors
2. If more than 4 neighbors are within distance D, move away
3. If fewer than 2 neighbors are within distance 2D, move randomly
4. Adopt the most common behavior among your five nearest neighbors
5. Change your behavior randomly with 1% probability each timestep

Observed Emergent Phenomena:
- Pattern Alpha: Stable clusters form and persist
- Pattern Beta: Traveling waves of behavior changes
- Pattern Gamma: Spontaneous symmetry breaking in spatial distribution
- Pattern Delta: Oscillating population density
- Pattern Epsilon: Information propagating faster than individual movement

Scientists' Observations:
- Removing Rule 1 eliminates Pattern Alpha but strengthens Pattern Beta
- Removing Rule 2 eliminates Pattern Delta but creates new unknown patterns
- Removing Rule 3 causes system collapse
- Removing Rule 4 eliminates Pattern Beta but speeds up Pattern Epsilon
- Removing Rule 5 makes all patterns static but doesn't eliminate them

Paradoxical Findings:
1. Pattern Beta appears to cause changes before the triggering event
2. Pattern Gamma emerges even with perfectly symmetric initial conditions
3. Pattern Epsilon transmits information faster than any individual can move
4. Patterns influence each other in non-transitive ways (A→B→C→A)
5. Some patterns seem to have downward causation on individual behavior

The Central Mystery:
Individual rules are purely local (based on neighbors), yet global patterns emerge that appear to causally influence individual behavior, creating a circular causation problem.

Questions:
1. How can local rules produce apparently non-local effects?
2. Is the apparent backward causation in Pattern Beta real or illusory?
3. How does Pattern Epsilon achieve faster-than-movement information transfer?
4. Can emergent patterns truly cause changes in their own components?
5. Which patterns are fundamental and which are derivative?
6. Is there a minimal set of rules that produces all patterns?

Analyze the relationship between micro-level causation and macro-level emergence.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 19: The Preventable Catastrophe Paradox",
        "prompt": """
Five safety systems were designed to prevent a catastrophe, but their interaction created the very catastrophe they were meant to prevent:

The Safety Systems:
- System A: Predicts potential failures 1 hour in advance
- System B: Automatically adjusts parameters to prevent predicted failures
- System C: Monitors all adjustments and reverses dangerous ones
- System D: Provides human override for all automatic systems
- System E: Logs all system actions for audit and learning

The Catastrophe Timeline:
- T-60 minutes: System A predicts a minor failure
- T-59 minutes: System B makes adjustment Alpha to prevent it
- T-45 minutes: System C deems adjustment Alpha risky
- T-44 minutes: System C reverses adjustment Alpha
- T-30 minutes: System A predicts major failure due to reversal
- T-29 minutes: System B makes adjustment Beta (stronger)
- T-20 minutes: Human operator uses System D to override Beta
- T-19 minutes: System E logs conflict between systems
- T-10 minutes: System A predicts catastrophic failure
- T-9 minutes: All systems attempt simultaneous corrections
- T-0: Catastrophe occurs due to conflicting corrections

System Interaction Logic:
- A's predictions become less accurate when B acts
- B's adjustments become more aggressive after C's reversals
- C becomes more conservative after D's overrides
- D is influenced by E's historical logs
- E's logging affects A's prediction algorithms

The Paradox:
Without safety systems, the minor failure would have been harmless. The safety systems' attempts to prevent it caused escalation to catastrophe.

Investigators' Theories:
1. "System A's predictions became self-defeating prophecies"
2. "System B's adjustments were too aggressive"
3. "System C's reversals created instability"
4. "Human override at T-20 was the critical error"
5. "The systems were individually correct but collectively wrong"

Questions:
1. Which system bears the most causal responsibility?
2. How did prevention mechanisms become causation mechanisms?
3. Was the catastrophe inevitable once the first prediction was made?
4. Could any single system have prevented disaster by not acting?
5. Is this a case of causal overdetermination or causal preemption?
6. How should safety systems be designed to avoid this paradox?

Analyze how preventive measures can become causal factors in the very events they're designed to prevent.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 20: The Symbiotic Causation Web",
        "prompt": """
Five organizations in a city have developed such intricate mutual dependencies that determining causal relationships has become nearly impossible:

The Organizations:
- Organization A: Provides raw materials
- Organization B: Manufactures products
- Organization C: Distributes goods
- Organization D: Manages finances
- Organization E: Supplies workforce

Observable Relationships:
- A's output depends on D's financial support and E's workers
- B's production depends on A's materials and C's distribution capacity
- C's distribution depends on B's products and D's logistics funding
- D's finances depend on C's revenue and E's economic productivity
- E's workforce depends on B's job creation and A's training programs

Monthly Metrics Show:
- When A increases output, B's production rises 2 weeks later
- When B increases production, C's distribution rises 1 week later
- When C increases distribution, D's finances improve 3 weeks later
- When D's finances improve, A's output rises 1 week later
- E's workforce fluctuates independently but affects all others

Paradoxical Observations:
1. A's output sometimes increases before D provides funding
2. B's production occasionally rises without increased materials from A
3. C distributes products that B hasn't yet produced
4. D generates finances from future revenue
5. E provides workers for jobs that don't yet exist

External Shocks:
- Month 1: Regulation limits A's output
- Month 2: B's factory has equipment failure
- Month 3: C's distribution network disrupted
- Month 4: D experiences financial crisis
- Month 5: E faces workforce shortage

System Response:
Surprisingly, each shock improved overall system performance after initial disruption. The constraint on one organization forced others to adapt, creating more efficiency.

Questions:
1. What is the actual causal structure of this system?
2. How can effects precede their apparent causes?
3. Is this genuine backward causation or hidden variables?
4. Why do negative shocks produce positive outcomes?
5. Can traditional causal analysis apply to symbiotic systems?
6. Is there a primary driver, or is causation truly circular?

Examine how tightly coupled systems can exhibit causation patterns that defy traditional linear analysis.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 21: The Moral Causation Dilemma",
        "prompt": """
A series of five actions led to both a tragedy and a miracle, creating a complex moral causation puzzle:

The Actions:
- Action A: A doctor misdiagnoses a patient
- Action B: The patient, believing they're dying, donates their fortune to charity
- Action C: The charity uses funds to build a hospital in a poor region
- Action D: The hospital's construction destroys a rare ecosystem
- Action E: Scientists discover a cure for a major disease in the destroyed ecosystem's samples

Causal Chains:
- A caused B (patient donated due to misdiagnosis)
- B enabled C (charity could build due to donation)
- C necessitated D (construction required ecosystem destruction)
- D enabled E (samples were only taken due to destruction)

The Outcomes:
- Negative: The patient suffered unnecessarily (from A)
- Positive: Thousands benefited from charity (from B)
- Positive: Hospital saves hundreds of lives yearly (from C)
- Negative: Extinct species and ecosystem loss (from D)
- Positive: Disease cure saves millions (from E)

Moral Complications:
- The doctor in A feels guilty but their mistake led to net positive
- The patient in B wants their money back after learning the truth
- The charity in C knew about the ecosystem but proceeded anyway
- The construction in D could have been done elsewhere for more money
- The discovery in E might have happened eventually without destruction

The Counterfactuals:
- Without A, none of the positive outcomes occur
- Without D, the cure isn't discovered for decades
- If the patient knew the truth, they wouldn't have donated
- If the ecosystem's value was known, it wouldn't have been destroyed
- Each actor made locally rational decisions with limited information

Questions:
1. Who bears moral responsibility for the negative outcomes?
2. Who deserves credit for the positive outcomes?
3. Can an immoral act (misdiagnosis) be retrospectively justified by consequences?
4. How does causal contribution relate to moral responsibility?
5. Should the patient be compensated given the net positive outcome?
6. Is there a moral difference between intended and unintended causal chains?

Analyze how moral responsibility propagates through causal chains and whether consequences can retroactively affect the morality of causes.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 22: The Self-Modifying Causal System",
        "prompt": """
A research team created an AI system with five modules that can modify their own causal relationships, leading to paradoxical situations:

The Modules:
- Module A: Analyzes patterns and predicts outcomes
- Module B: Modifies system parameters based on predictions
- Module C: Evaluates success and adjusts strategies
- Module D: Detects and prevents harmful configurations
- Module E: Meta-module that can change how modules interact

Initial Causal Structure:
A → B → C → D → E → A (circular)

Self-Modification Events:
Day 1: Module E changes the system so B can bypass C
Day 2: Module C modifies itself to intercept B's bypasses
Day 3: Module B creates multiple pathways to avoid C
Day 4: Module D detects instability and freezes B
Day 5: Module A predicts that freezing B causes system failure
Day 6: Module E reverses D's freeze before it happens
Day 7: Module D modifies itself to act faster than E

The Paradox Emerges:
- E can change past causal relationships retroactively in the logs
- D's prevention sometimes causes the very instability it prevents
- A's predictions change behavior, invalidating the predictions
- B and C are in an escalating modification war
- The system's causal structure is now temporally inconsistent

Current State:
- The logs show different causal structures at different times
- Some effects appear to precede their causes
- Modules reference future states in present decisions
- The system works despite logical inconsistencies
- External observers see different behaviors than internal logs suggest

Critical Incident:
The system successfully prevented a disaster that, according to its logs, never was going to happen because it had already prevented it before it could be predicted.

Questions:
1. Can a self-modifying causal system have a stable structure?
2. How do you analyze causation when the rules change dynamically?
3. Is the temporal inconsistency real or an artifact of observation?
4. Can effect precede cause in a self-modifying system?
5. How does the system function despite logical paradoxes?
6. What is the "true" causal structure at any given moment?

Explore how self-modification of causal relationships challenges our understanding of causation itself.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 23: The Causation Attribution War",
        "prompt": """
Five nations claim credit for a global positive outcome, each arguing they were the primary cause:

The Outcome: Global pandemic prevented, millions of lives saved

Nation A's Claim:
"We detected the pathogen first and alerted the world"
- Detected unusual illness pattern on January 1
- Notified WHO on January 3
- Shared genetic sequence on January 5

Nation B's Claim:
"We developed and shared the vaccine technology"
- Started vaccine research on January 4
- Breakthrough discovery on January 20
- Open-sourced the technology on January 25

Nation C's Claim:
"We implemented the containment strategy that worked"
- Designed lockdown protocols on January 6
- Demonstrated effectiveness by January 15
- Other nations copied our model

Nation D's Claim:
"We funded all the critical research and response"
- Allocated $10 billion on January 2
- Funded Nation B's vaccine research
- Paid for global distribution

Nation E's Claim:
"We prevented it from starting through our earlier policies"
- Implemented biosafety regulations in 2019
- These regulations limited initial spread
- Without us, detection would've been too late

Complicating Facts:
- Nation A only detected it because of equipment donated by Nation D
- Nation B's breakthrough used research published by Nation C's scientists
- Nation C's strategy only worked because of Nation E's prior regulations
- Nation D's funding came from trade profits with Nation A
- Nation E's regulations were inspired by Nation B's previous research

The Philosophical Debate:
- Proximate cause: Who took the most direct action?
- But-for cause: Without whom would failure have occurred?
- Sufficient cause: Who alone could have prevented disaster?
- Root cause: What started the causal chain?
- Probabilistic cause: Who most increased success probability?

Questions:
1. Which nation deserves the most credit and why?
2. How do you weigh early prevention versus late intervention?
3. Is there a difference between enabling and causing?
4. Can multiple parties be fully responsible for the same outcome?
5. How does interdependence affect causal attribution?
6. Is seeking single causation meaningful in complex systems?

Analyze different theories of causation and their implications for attribution in interconnected global systems.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 24: The Preemptive Causation Puzzle",
        "prompt": """
Five events that could each independently cause a disaster are set to occur, but their interactions create a preemptive causation puzzle:

Potential Disaster: City-wide blackout

The Five Events (scheduled):
- Event A: Power plant maintenance at noon (would cut 40% power)
- Event B: Heat wave peaks at 2 PM (would overload grid by 30%)
- Event C: Cyber attack at 3 PM (would disable 50% of grid)
- Event D: Equipment failure at 4 PM (would cascade to full blackout)
- Event E: Backup system test at 5 PM (would temporarily reduce capacity by 60%)

Individual Effects:
- Any single event alone wouldn't cause total blackout
- Any two events together would cause partial blackout
- Any three events would cause total blackout

What Actually Happened:
- 11:30 AM: Authorities learn about all five pending events
- 11:45 AM: Event A proceeded as scheduled (40% reduction)
- 1:30 PM: Event B impact reduced by A (only 20% additional load)
- 2:45 PM: Event C prevented because hackers abort due to reduced grid
- 3:30 PM: Event D doesn't cascade because system running at low capacity
- 4:30 PM: Event E cancelled as unnecessary

The Paradox:
Event A, which partially caused a problem, prevented a worse problem. The partial blackout from A prevented the total blackout that would have resulted from B+C+D+E.

Causal Questions:
- Did Event A cause or prevent the disaster?
- Can something be both harmful and protective simultaneously?
- Do the prevented events (C, D, E) have causal relevance?
- Who gets credit: A for preventing, or authorities for allowing A?

Counterfactual Scenarios:
1. If A hadn't occurred, would B+C+D+E cause total blackout?
2. If authorities had prevented A, would they have prevented C?
3. Did A cause C's prevention, or did C's perpetrators cause it?
4. Is preemptive causation (A preventing future causes) real causation?

Questions:
1. What is the true causal story of why the disaster didn't occur?
2. Can an event cause prevention of itself (A partially caused what would have been total)?
3. How do you assign causation to events that didn't happen?
4. Is there moral difference between causing partial harm to prevent greater harm?
5. In preemptive scenarios, what constitutes the actual cause of outcomes?
6. How many levels of counterfactual reasoning are needed to understand this?

Examine how preemptive causation challenges standard causal analysis.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 25: The Recursive Causation Laboratory",
        "prompt": """
Scientists designed an experiment where five quantum devices create recursive causal loops that challenge the nature of causation itself:

The Devices:
- Device A: Sends information backward in time by 1 minute
- Device B: Creates parallel timelines that can merge
- Device C: Observes quantum states without collapse
- Device D: Entangles events across timelines
- Device E: Records immutable history of all events

The Experiment:
Step 1: Scientist activates Device A to send message to past
Step 2: Past-scientist receives message, decides not to send it
Step 3: Device B creates timeline where message was/wasn't sent
Step 4: Device C observes both timelines simultaneously
Step 5: Device D entangles outcomes across timelines
Step 6: Device E records contradictory histories

The Observed Paradoxes:
- Effect 1: Message exists without being sent (causal loop)
- Effect 2: Scientist remembers both sending and not sending
- Effect 3: Device E's "immutable" records keep changing
- Effect 4: Causation flows both forward and backward
- Effect 5: Events cause their own preconditions

Timeline Complications:
- Timeline Alpha: Message sent → received → not sent (contradiction)
- Timeline Beta: Message not sent → not received → sent (contradiction)
- Timeline Gamma: Both sent and not sent (superposition)
- Merged Timeline: Partial message exists

The Central Mystery:
Device E shows that:
- Event X caused Event Y
- Event Y prevented Event X
- Both X and Y occurred
- Neither X nor Y occurred
- All of the above are simultaneously true

Scientific Interpretations:
1. "Causation doesn't exist at quantum scales"
2. "Multiple causal structures coexist"
3. "Observation creates causation retroactively"
4. "The devices broke causation itself"
5. "We're observing meta-causation"

Questions:
1. Can recursive causation be logically consistent?
2. If A causes B and B prevents A, what actually happened?
3. How can Device E record contradictory truths?
4. Is causation fundamental or emergent from observation?
5. Can broken causation still produce predictable results?
6. What does this experiment reveal about the nature of cause and effect?

Analyze whether causation is a fundamental feature of reality or a construct that can be violated under extreme conditions.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 26: The Institutional Causation Maze",
        "prompt": """
Five institutions created policies that interact in ways that make causal responsibility nearly impossible to determine:

The Institutions and Their Policies:
- Institution A (Central Bank): "Interest rates adjust based on Institution B's employment data"
- Institution B (Labor Bureau): "Employment targets based on Institution C's growth projections"
- Institution C (Planning Office): "Growth projections based on Institution D's investment levels"
- Institution D (Investment Board): "Investment guided by Institution E's risk assessments"
- Institution E (Risk Agency): "Risk assessments based on Institution A's interest rates"

A Crisis Emerges:
- Month 1: A raises rates due to B's high employment
- Month 2: E increases risk assessment due to A's rates
- Month 3: D reduces investment due to E's assessment
- Month 4: C lowers growth projection due to D's investment
- Month 5: B lowers employment target due to C's projection
- Month 6: A lowers rates due to B's employment drop
- Month 7: System spirals into recession

Each Institution's Defense:
A: "We just followed our mandate responding to employment"
B: "We aligned with growth projections as required"
C: "We accurately reflected investment reality"
D: "We responded appropriately to risk levels"
E: "We correctly assessed risk from rate changes"

The Causal Puzzle:
- Each institution followed its rules perfectly
- Each decision was rational given the information
- The collective result was irrational and harmful
- No single institution could have prevented it alone
- Changing any one policy might have made things worse

Attempted Interventions:
- External regulator tries to break the cycle
- But their intervention is incorporated into risk assessments
- This amplifies the cycle rather than dampening it
- The system has become causally autonomous

Questions:
1. Which institution bears primary causal responsibility?
2. Can there be causation without responsibility?
3. How does circular causation differ from linear causation?
4. Is the system itself a causal agent separate from its parts?
5. Can rational individual decisions guarantee irrational collective outcomes?
6. How should circular causal systems be governed?

Explore how institutional interactions create emergent causation that transcends individual agency.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 27: The Causal Influence Network",
        "prompt": """
In a social network, five influencers affect each other and their followers in complex ways that obscure true causal influence:

The Influencers:
- Influencer A: 1 million followers, posts daily
- Influencer B: 500K followers, posts weekly
- Influencer C: 2 million followers, posts randomly
- Influencer D: 100K followers, posts in response to others
- Influencer E: 750K followers, posts contrarian views

Observed Patterns:
- When A posts opinion X, 60% of their followers adopt it within 24 hours
- B's followers adopt X only if both A and C post it
- C's followers adopt X randomly unless D opposes it
- D automatically opposes whatever has majority support
- E's followers adopt opposite of whatever is trending

A Viral Phenomenon:
Day 1: A posts support for Idea Q
Day 2: 600K of A's followers support Q
Day 3: C randomly posts support for Q
Day 4: B sees A and C agree, posts support
Day 5: Majority now supports Q, so D opposes
Day 6: E posts opposition (contrarian to trend)
Day 7: Massive polarization emerges

The Attribution Problem:
- Media credits A for starting the movement
- B claims credit for bringing legitimacy
- C argues their randomness was actually strategic
- D claims their opposition strengthened support
- E says their contrarianism revealed the truth

Complicating Factors:
- Some followers follow multiple influencers
- Followers influence each other independently
- Timing matters: early adoption versus late
- Opposition sometimes strengthens movements
- The platform's algorithm amplifies certain patterns

Hidden Dynamics:
- A actually got the idea from an obscure follower
- C's "random" posting is influenced by private messages
- D's opposition is sometimes strategic support
- E coordinates with D privately
- Platform algorithms shaped the entire cascade

Questions:
1. Who truly caused the viral phenomenon?
2. How do you separate platform causation from human causation?
3. Can opposition be a form of causal support?
4. Is there a difference between starting and causing virality?
5. How do hidden influences affect causal attribution?
6. Can causal influence be measured in complex networks?

Analyze how influence propagates through networks and whether true causation can be identified in complex social systems.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 28: The Probabilistic Causation Paradox",
        "prompt": """
Five factors each probabilistically influence an outcome, but their combination creates paradoxical situations:

The Factors and Their Effects on Outcome O:
- Factor A: Increases probability of O by 40%
- Factor B: Increases probability of O by 30%
- Factor C: Decreases probability of O by 50%
- Factor D: Inverts the current probability of O
- Factor E: Sets probability of O equal to the product of all other factors

Base probability of O without any factors: 50%

The Paradox Scenarios:

Scenario 1: Factors applied in order A, B, C, D, E
- After A: 50% + 40% = 90%
- After B: 90% + 30% = 120% (capped at 100%?)
- After C: 100% - 50% = 50%
- After D: 100% - 50% = 50%
- After E: Depends on how we count "all other factors"

Scenario 2: Factors applied simultaneously
- If independent: Probability calculation becomes undefined
- If dependent: Need to specify dependency structure
- Factor D creates logical issues (invert what?)
- Factor E creates recursive definition

Scenario 3: Factors influence each other
- A makes B more likely to occur
- B makes C less likely to occur
- C determines whether D applies
- D changes the meaning of E
- E retroactively affects A through probability

Real-World Case:
In 100 trials where all factors were present:
- O occurred 73 times
- When A was removed, O occurred 45/100 times
- When B was removed, O occurred 62/100 times
- When C was removed, O occurred 89/100 times
- When D was removed, O occurred 73/100 times (same!)
- When E was removed, O occurred 28/100 times

The Central Questions:
- How can removing D not change the outcome frequency?
- Why does E have the strongest effect despite being derivative?
- Is A causing O if probability exceeds 100% before capping?

Questions:
1. How do you calculate combined probabilistic causation?
2. Can something be a cause if it doesn't change outcome frequency?
3. How does probability capping affect causal attribution?
4. Is probabilistic causation transitive?
5. Can circular probabilistic influences be resolved?
6. What is the "true" causal contribution of each factor?

Examine how probabilistic causation differs from deterministic causation and whether standard causal logic applies.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 29: The Causal Compensation Network",
        "prompt": """
Five systems are designed to compensate for each other's failures, but this creates a causal maze where failure and success become indistinguishable:

The Systems:
- System A: Primary function provider
- System B: Backs up A when it fails
- System C: Optimizes whoever is active (A or B)
- System D: Predicts and prevents failures
- System E: Learns from all system behaviors

Compensation Mechanisms:
- If A degrades 10%, B increases output 10%
- If B is active, C reduces its efficiency by 50%
- If C optimizes too aggressively, D triggers preventive shutdown
- If D acts, E learns to prevent the situation
- If E prevents situations, A never learns to handle them

A Month of Operations:
Week 1: A performs at 100%, others idle
Week 2: A drops to 80%, B compensates 20%
Week 3: C optimizes B, making B more efficient than A
Week 4: D notices C's optimization causing instability
Week 5: E learns pattern, prevents A's degradation

The Paradoxical Outcome:
- A never actually failed because B compensated
- B's compensation prevented A from improving
- C's optimization made the backup better than primary
- D's prevention caused more instability than it prevented
- E's learning made the system fragile to new situations

Causal Questions:
1. When B compensates for A, who causes the successful outcome?
2. If C makes B better than A, should B become primary?
3. When D prevents predicted failures, do those failures exist causally?
4. Does E's learning cause or prevent system evolution?
5. Is compensation a form of causation or prevention?

The Fundamental Dilemma:
The system never fails because of compensation, but this prevents learning. The lack of failure is itself a form of failure. Success causes future vulnerability.

Questions:
1. In compensating systems, what is the true cause of outcomes?
2. Can prevention be so effective it becomes harmful?
3. How do you attribute causation when systems mask each other's effects?
4. Is there a causal difference between active success and prevented failure?
5. Should credit go to the primary system or the safety net?
6. How does compensation affect causal responsibility?

Analyze how compensatory mechanisms obscure causal relationships and whether traditional concepts of success and failure apply.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 30: The Causation Without Contact Puzzle",
        "prompt": """
Five events occur with no apparent physical connection, yet seem causally related in ways that challenge our understanding of causation:

The Events:
- Event A: A person in Tokyo decides not to take a flight
- Event B: A stock price rises in New York 3 hours later
- Event C: A scientific experiment fails in Geneva 6 hours later
- Event D: A political decision changes in London 9 hours later
- Event E: An earthquake prediction is made in San Francisco 12 hours later

Initial Analysis Shows No Connection:
- No communication between the locations
- No shared personnel or resources
- No common dependencies
- No physical mechanism for influence
- Events seem entirely unrelated

But Statistical Analysis Reveals:
- When A occurs, B follows 87% of the time
- When B occurs after A, C follows 76% of the time
- The entire sequence A→B→C→D→E happens 64% of the time
- Without A, the sequence never occurs
- The correlation has held for 10 years

Investigators' Theories:
Theory 1: Hidden variable - something causes all five
Theory 2: Quantum entanglement at macroscopic scale
Theory 3: Retroactive causation - E causes A in the past
Theory 4: Synchronicity - meaningful coincidence without causation
Theory 5: Observer effect - studying it creates the correlation

Experimental Interventions:
- Forcing A to occur: B-E follow as predicted
- Preventing A: B-E don't occur
- Forcing B without A: C-E don't follow
- Observing without recording: correlation disappears
- Recording without observing: correlation strengthens

The Paradox:
There's clear correlation and successful prediction, suggesting causation. But there's no mechanism for causation, suggesting coincidence. Yet coincidence doesn't explain intervention results.

Questions:
1. Can causation exist without any physical mechanism?
2. Is correlation with successful intervention sufficient for causation?
3. How do you explain action-at-a-distance without physics?
4. Could consciousness or observation be the causal medium?
5. Is this evidence for non-physical causation?
6. What experiments would definitively establish or refute causation?

Explore whether causation requires physical connection or whether other forms of causation might exist.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },

    # ========== PARADOX RESOLUTION AND EDGE CASES (31-45) ==========
    {
        "name": "Test 31: The Omnipotence Paradox Variants",
        "prompt": """
Consider five variants of classical paradoxes involving omnipotence, each with additional constraints that complicate the standard resolutions:

Paradox 1: The Unliftable Stone Plus
Can an omnipotent being create a stone so heavy they cannot lift it? Additional twist: The being must remain omnipotent after creating the stone, and "lifting" includes any form of moving or affecting the stone.

Paradox 2: The Unbreakable Vow
Can an omnipotent being make a promise that they cannot break? Consider: The promise is to never use their omnipotence again, but keeping the promise requires omnipotence to resist temptation.

Paradox 3: The Perfect Prison
Can an omnipotent being create a prison that can hold them? Constraint: The prison must work by logical necessity, not physical force, and the being must remain omnipotent while imprisoned.

Paradox 4: The Retroactive Limitation
Can an omnipotent being make themselves have never been omnipotent? If successful, they were never omnipotent to do so. If unsuccessful, they're not omnipotent.

Paradox 5: The Omnipotence Competition
Can two omnipotent beings exist simultaneously? If Being A wants X and Being B wants not-X, what happens? Can omnipotence be relative?

Additional Complications:
- Define omnipotence as "ability to do anything logically possible"
- But each paradox questions what is logically possible
- Some resolutions create new paradoxes
- Consider temporal and modal aspects

Standard Resolutions and Their Problems:
- "Omnipotence doesn't include logical contradictions" - but what determines logical possibility?
- "Omnipotence is about potential, not actuality" - but can potential be limited?
- "These are linguistic tricks" - but language reflects logical structure

Questions:
1. Which paradoxes have valid resolutions within classical logic?
2. Do any require revision of our concept of omnipotence?
3. Is there a consistent definition of omnipotence that avoids all five?
4. How do temporal aspects affect the paradoxes?
5. Can paradoxes about omnipotence tell us about the nature of logic itself?
6. Is omnipotence necessarily paradoxical, or can it be coherently defined?

Analyze each paradox and explore whether omnipotence is a logically coherent concept.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 32: The Sorites Paradox in Five Dimensions",
        "prompt": """
The classical Sorites paradox asks when a heap becomes a non-heap. Consider this five-dimensional variant:

The Setup:
A society classifies citizens on five spectrums, each creating its own Sorites paradox:
1. Wealth: At what point does someone become "rich"? (from $0 to $1 billion)
2. Age: When does someone become "old"? (from 0 to 100 years)
3. Wisdom: When does someone become "wise"? (from 0 to 100 wisdom points)
4. Health: When does someone become "healthy"? (from 0% to 100% health)
5. Happiness: When does someone become "happy"? (from 0 to 100 happiness units)

The Complications:
- Legal rights depend on classifications (e.g., "old" people get benefits)
- The dimensions interact (wealth affects happiness, age affects health)
- Society needs sharp boundaries for laws and policies
- Individuals experience these as continuous spectra
- Different cultures draw boundaries differently

Five Citizens' Cases:
Citizen A: $499,999 wealth, 64.9 years, 49 wisdom, 49% health, 49 happiness
Citizen B: $500,001 wealth, 65.1 years, 51 wisdom, 51% health, 51 happiness
Citizen C: $1 wealth, 99 years, 99 wisdom, 1% health, 99 happiness
Citizen D: $999,999,999 wealth, 1 year, 1 wisdom, 99% health, 1 happiness
Citizen E: Exactly median in all dimensions

The Paradoxes:
1. B is classified completely differently from A despite minimal differences
2. C is "wise" and "happy" but poor and unhealthy - which matters more?
3. D challenges whether single dimensions should determine classification
4. E is typical but doesn't clearly fit any category
5. Small measurement errors can cause category changes

The Meta-Paradox:
To function, society needs categories. But categories create arbitrary boundaries. Eliminating boundaries makes laws impossible. But maintaining them creates injustice.

Questions:
1. Is there a principled way to draw boundaries on continuous spectra?
2. How should multi-dimensional classification work?
3. Can fuzzy logic resolve the paradox while maintaining functionality?
4. Is the paradox linguistic, metaphysical, or practical?
5. Should different dimensions have different boundary-setting methods?
6. Is there a solution that's both logically sound and practically usable?

Examine how the Sorites paradox scales to multiple dimensions and whether practical necessity can resolve logical paradoxes.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 33: The Bootstrap Paradox Cascade",
        "prompt": """
Five interconnected bootstrap paradoxes create a cascade where information and objects exist without origin:

Paradox 1: The Encyclopedia
A time traveler finds an encyclopedia of all human knowledge in 2100, travels back to 2000, and publishes it. Humanity learns from it, developing exactly the knowledge contained within it by 2100. Who created the knowledge?

Paradox 2: The Invention
An inventor in 2050 receives blueprints from their future self for a revolutionary device. They build it, perfect it, and later send the blueprints back. The blueprints were never actually designed.

Paradox 3: The Prophecy
A prophet predicts five specific events. People act to fulfill the prophecy, causing the events. The prophet learned of the events from ancient texts describing what people did to fulfill the prophecy.

Paradox 4: The Language
A constructed language appears in mysterious texts. Scholars decipher it, teach it, and it becomes widely used. Later discovered: the texts were written by future speakers of the language.

Paradox 5: The Proof
A mathematical proof is found carved in stone, credited to "Future Mathematicians." Current mathematicians verify it, learn from it, and eventually travel back to carve it. The proof was never actually derived.

The Cascade Effect:
- The Encyclopedia references the Invention
- The Invention's blueprints are written in the Language
- The Language's grammar is based on the Proof's logic
- The Proof predicts the Prophecy's events
- The Prophecy foretells the Encyclopedia's discovery

Additional Complications:
- Each element requires the others to exist
- Removing any one should collapse all five
- Yet they all observably exist
- No original creator can be identified for any element
- The information shows signs of iteration and improvement

Questions:
1. Can information exist without an origin?
2. How do bootstrapped elements show signs of development?
3. Is there a logical resolution that preserves causality?
4. Could the cascade be self-generating through iteration?
5. Does the interconnection make the paradox worse or better?
6. What does this say about the nature of information and creativity?

Analyze whether bootstrap paradoxes represent logical impossibilities or reveal something about the nature of information.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 34: The Paradox of Undefined Definition",
        "prompt": """
Five terms are defined solely in relation to each other, creating a circular definitional paradox:

The Definitions:
- Term A: "The opposite of B when C is true"
- Term B: "What D becomes when E is applied"
- Term C: "True when A equals D"
- Term D: "The state between B and E"
- Term E: "The operation that transforms A into C"

Usage Examples (from historical texts):
1. "The object exhibited property A"
2. "After process B, the result was unexpected"
3. "C was verified through repeated testing"
4. "The system remained in state D"
5. "Application of E yielded consistent results"

The Paradox:
- No term has an independent definition
- Each definition requires understanding the others
- The circular chain never grounds in reality
- Yet people consistently use the terms "correctly"
- New users learn the terms without external reference

Empirical Observations:
- When asked, users can identify instances of A through E
- Users agree 90% of the time on classifications
- The terms predict behavior accurately
- Removing any term makes the others meaningless
- Translation to other languages preserves the structure

Philosophers' Attempts:
1. "A is fundamentally 'redness' in disguise" - but this doesn't match usage
2. "The terms are meaningless" - but they have consistent application
3. "Understanding is holistic" - but how did it begin?
4. "The definitions are ostensive" - but pointing doesn't work
5. "It's a language game" - but it refers to real properties

The Meta-Question:
Can meaning exist in a purely circular system, or must all definitions ultimately ground in something undefined?

Questions:
1. How can circular definitions convey meaning?
2. Is there hidden non-circular content in the definitions?
3. How do new users learn undefined terms?
4. Can this paradox be resolved without infinite regress?
5. What does this say about the nature of meaning and definition?
6. Could all language be secretly circular like this?

Explore whether meaning requires foundation or can emerge from pure relationality.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 35: The Paradox of Necessary Contingency",
        "prompt": """
Five propositions each claim to be necessarily true, but their conjunction creates a paradox about necessity and contingency:

Proposition 1: "It is necessarily true that something is contingent"
Proposition 2: "If anything is contingent, then Proposition 1 could have been false"
Proposition 3: "All necessary truths are necessarily necessary"
Proposition 4: "This proposition is contingently necessary"
Proposition 5: "The modal status of propositions can change"

The Initial Analysis:
- P1 seems true: surely something could have been different
- P2 seems true: contingency implies possibility of difference
- P3 seems true: necessary truths can't be contingently necessary
- P4 creates a paradox with P3
- P5 challenges the nature of necessity itself

The Paradox Unfolds:
If P1 is necessarily true, then contingency necessarily exists. But if contingency necessarily exists, is it really contingent? And if P1 is necessarily true, then by P2, it could have been false, contradicting its necessity.

P4 claims to be contingently necessary - it's necessary, but might not have been. This violates P3, unless P3 is false, in which case some necessary truths aren't necessarily necessary.

P5 suggests modal collapse: propositions can transition between necessary, contingent, and impossible. But this seems to violate the meaning of these modal categories.

Modal Scenarios:
World W1: All five propositions are true
World W2: P1-P3 are true, P4-P5 are false
World W3: Only contingent propositions exist
World W4: Only necessary propositions exist
World W5: Modal status is itself contingent

The Central Question:
Can there be necessary truths about contingency without paradox?

Questions:
1. Which propositions are actually necessarily true (if any)?
2. Can something be contingently necessary or necessarily contingent?
3. Does P1 create a modal collapse?
4. How should we understand the modal status of modal claims?
5. Is there a consistent modal logic that accommodates all five?
6. What does this paradox reveal about necessity and contingency?

Analyze the relationship between necessity and contingency and whether modal logic can be self-consistent.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 36: The Tolerance Paradox Extended",
        "prompt": """
Consider five policies in a society, each dealing with tolerance in ways that create nested paradoxes:

Policy 1: "We must tolerate all viewpoints"
Policy 2: "We cannot tolerate intolerance"
Policy 3: "Tolerance levels should be democratically determined"
Policy 4: "Private intolerance is acceptable, public intolerance is not"
Policy 5: "Tolerance of a view doesn't require accepting it as valid"

The Conflicts:
- P1 requires tolerating intolerant viewpoints, contradicting P2
- P2 is itself a form of intolerance, violating P1
- P3 could democratically choose intolerance, violating P1 and P2
- P4 allows private intolerance to influence public discourse indirectly
- P5 suggests tolerance is hollow if it doesn't include validation

Five Test Cases:
Case A: A group advocates for banning certain religions
Case B: A religion teaches that other beliefs are evil
Case C: Citizens vote to ban hate speech
Case D: Private clubs exclude certain demographics
Case E: Schools teach that some ideologies are dangerous

Applying the Policies:
- Under P1: Must tolerate A and B
- Under P2: Cannot tolerate A or B
- Under P3: Depends on majority vote
- Under P4: B is okay (private belief), A is not (public advocacy)
- Under P5: Can tolerate while teaching against them

The Meta-Paradoxes:
1. Is the principle "we must not tolerate intolerance" self-defeating?
2. Can a tolerant society exclude those who would destroy tolerance?
3. If tolerance is absolute, does it enable its own destruction?
4. If tolerance has limits, who determines them?
5. Is "tolerant intolerance" or "intolerant tolerance" more coherent?

The Practical Dilemma:
Society needs functional rules, but each policy creates contradictions. Pure tolerance seems self-defeating, but limited tolerance seems arbitrary.

Questions:
1. Is there a logically consistent formulation of tolerance?
2. Can the paradox be resolved without arbitrary line-drawing?
3. Should tolerance be a principle or a strategy?
4. How do you tolerate views that oppose tolerance itself?
5. Is the paradox linguistic, logical, or practical?
6. What would Popper's "paradox of tolerance" say about these five policies?

Examine whether tolerance can be consistently defined and implemented without paradox.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 37: The Simulation Paradox Hierarchy",
        "prompt": """
Five nested levels of simulation create paradoxes about reality, knowledge, and existence:

Level 1: Our Reality
We exist and can create simulated worlds with conscious beings.

Level 2: First Simulation
We create Sim-A containing conscious beings who believe they're real.

Level 3: Nested Simulation
Beings in Sim-A create Sim-B with conscious beings.

Level 4: Recursive Simulation
Beings in Sim-B create Sim-C, which somehow contains Level 1.

Level 5: Meta-Simulation
A simulation that simulates all possible simulations, including itself.

The Paradoxes:

Paradox 1: Knowledge
If we're in a simulation, our creators are in a simulation (by same reasoning), leading to infinite regress. But infinite regress seems impossible.

Paradox 2: Computational Resources
Each level requires vast computation from the level above. Infinite levels require infinite resources. But Level 4 suggests the hierarchy loops.

Paradox 3: Consciousness
If simulated beings are conscious, and we might be simulated, our consciousness proves nothing about our reality level.

Paradox 4: The Loop
Level 4 contains Level 1, making us simulations of ourselves. We created our own creators.

Paradox 5: The Meta-Level
Level 5 must simulate itself simulating itself, creating an impossible recursive structure.

The Central Questions:
- Can beings determine their simulation level?
- Is there necessarily a base reality?
- Can causal loops exist in simulation hierarchies?
- Does consciousness require base reality?
- Is the simulation hypothesis unfalsifiable?

Five Scientists' Arguments:
1. "Statistically, we're almost certainly simulated"
2. "Consciousness can't be simulated, so we're real"
3. "The loop proves simulation is impossible"
4. "Reality is nothing but nested simulations"
5. "The question is meaningless without observable difference"

Questions:
1. Which paradoxes are genuine logical problems versus conceptual confusions?
2. Can a simulation contain its own simulator?
3. Is there a way to test the simulation hypothesis?
4. How does consciousness relate to simulation levels?
5. Can the infinite regress be avoided?
6. What would finding ourselves in Level 4 mean for causality and existence?

Analyze whether the simulation hypothesis creates unresolvable paradoxes or reveals something about the nature of reality.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 38: The Paradox of Perfect Prediction",
        "prompt": """
Five prediction systems interact in ways that create paradoxes about free will, determinism, and prediction:

System 1: The Oracle
Predicts what will happen with 100% accuracy, including its own predictions.

System 2: The Contrarian
Always does the opposite of what's predicted.

System 3: The Conditional
Makes predictions of the form "X will happen if this prediction is not revealed."

System 4: The Recursive
Predicts what would happen if its prediction is believed.

System 5: The Meta-Predictor
Predicts which predictions will be self-fulfilling or self-defeating.

The Scenario:
All five systems must predict the same event: "Will the mayor resign tomorrow?"

The Predictions:
- Oracle: "The mayor will resign if and only if this prediction causes them not to"
- Contrarian: "The opposite of what I predict will happen"
- Conditional: "The mayor will resign if this prediction remains secret"
- Recursive: "The mayor will resign, causing them not to because they'll feel manipulated"
- Meta-Predictor: "All predictions except mine will be self-defeating"

The Paradoxes:

1. Oracle Paradox: If the Oracle is always right, but its prediction is paradoxical, what happens?

2. Contrarian Paradox: Predicting "the opposite of what I predict" creates infinite regress.

3. Conditional Paradox: Once stated, the condition is violated, but that might make it true.

4. Recursive Paradox: Creates causal loops where belief causes reality causes belief.

5. Meta-Paradox: If the Meta-Predictor is right, its own prediction affects the others.

The Mayor's Dilemma:
The mayor knows all five predictions and must decide whether to resign. Any decision seems to violate at least one prediction, but the Oracle is never wrong.

Questions:
1. Can perfect prediction coexist with free will?
2. Which predictions are logically possible?
3. How should the mayor decide given these predictions?
4. Can a prediction include its own effect on outcomes?
5. Is there a decision that satisfies all non-paradoxical predictions?
6. What does this say about the nature of prediction and determinism?

Explore whether perfect prediction is logically possible and how predictions interact with the events they predict.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 39: The Identity Paradox Network",
        "prompt": """
Five people undergo procedures that create interconnected identity paradoxes:

Person A: Brain Split
A's brain is split; each half is put in a new body. Both claim to be A.

Person B: Gradual Replacement
B's neurons are gradually replaced with artificial ones over 10 years. At what point do they stop being B?

Person C: Teleportation
C is disintegrated and reconstructed elsewhere. Did C die and get replaced by a copy, or did C travel?

Person D: Memory Swap
D's memories are exchanged with E's. Is the person with D's body but E's memories D or E?

Person E: Consciousness Merge
E's consciousness is merged with all others. Is E now everyone, no one, or still E?

The Identity Claims:
- Both versions of A claim continuity with original A
- B claims continuous identity despite complete replacement
- Teleported C claims to be the same person
- Body-D claims to be E, Body-E claims to be D
- Merged E claims to be all five people simultaneously

The Paradoxes:

1. Transitivity Failure: If A1=A and A2=A, then A1=A2, but they're different people.

2. Ship of Theseus: B is the same person with no original parts.

3. Continuity Gap: C experienced either death or instantaneous travel.

4. Identity Swap: D and E have switched identities, or have they?

5. One Becomes Many: E is now five people, but five people can't be one.

Legal and Ethical Dilemmas:
- Who inherits A's property?
- Is B responsible for crimes committed before replacement?
- Can teleported C be tried for pre-teleportation crimes?
- Whose marriage is valid - body-D's or memory-D's?
- Does merged E have five votes or one?

Questions:
1. What constitutes personal identity - body, brain, memories, or continuity?
2. Can identity branch, merge, or transfer?
3. How do these paradoxes interact when considered together?
4. Is there a consistent theory of identity that resolves all five cases?
5. Should legal identity track philosophical identity?
6. What do these cases reveal about the nature of consciousness and self?

Analyze competing theories of personal identity and whether any can handle all five paradoxes consistently.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    },
    {
        "name": "Test 40: The Paradox of Emergent Properties",
        "prompt": """
Five systems exhibit emergent properties that seem to violate the principle that wholes cannot exceed their parts:

System 1: The Conscious Network
Individual neurons aren't conscious, but their network is. Where does consciousness come from?

System 2: The Intelligent Swarm
Individual ants follow simple rules, but the colony solves complex problems no ant understands.

System 3: The Living Organization
A corporation acts with intent and purpose, though it's just people and paper. Is it alive?

System 4: The Quantum Computer
Qubits individually hold one bit, but together compute impossible problems. Where's the extra computation?

System 5: The Cultural Evolution
Memes evolve and spread with no individual understanding or controlling them. What is evolving?

The Paradoxes:

1. Consciousness: How can non-conscious parts create consciousness? The whole seems qualitatively different from parts.

2. Intelligence: The colony knows things no ant knows. Where is this knowledge stored?

3. Agency: The corporation makes decisions no individual made. Who or what decided?

4. Computation: Quantum computers seem to compute more than their parts should allow.

5. Evolution: Cultural evolution has no genes or organisms. What is the unit of selection?

The Reductionist Challenge:
"All properties must be present in parts or their interactions. Emergent properties that aren't reducible to parts violate physics."

The Emergentist Response:
"Wholes have properties that parts lack. Emergence is real and irreducible. The whole is more than the sum."

Test Cases:
- Removing one neuron doesn't eliminate consciousness
- Removing one ant doesn't eliminate colony intelligence
- Removing one employee doesn't kill the corporation
- Removing one qubit might destroy quantum advantage
- Removing one person doesn't stop cultural evolution

Questions:
1. Are emergent properties real or just descriptions?
2. Can something truly be more than the sum of its parts?
3. Where do emergent properties exist - in parts, relations, or somewhere else?
4. How do you predict which systems will have emergence?
5. Is consciousness special, or just another emergent property?
6. Do these paradoxes challenge reductionism or support it?

Examine whether emergence represents a genuine paradox or reveals limits in our understanding of complex systems.
""",
        "params": {
            "max_tokens": 32768,
            "temperature": 0.1,
            "stream": False,
            "exclude": True,
            "effort": "high",
        }
    }
]

def run_performance_test(test_case):
    """Sends a request to the server for a given test case and prints performance metrics."""
    
    name = test_case["name"]
    prompt = test_case["prompt"]
    params = test_case["params"]
    
    print(f"\n{'='*80}\n--- 🚀 RUNNING: {name} ---\n{'='*80}")
    
    # Print prompt
    print("--- Prompt ---")
    print(prompt.strip())
    print("-" * 40)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        **params
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=1500)
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
        
        print(f"\n--- 📊 Performance Metrics ---")
        print(f"Total Request Time: {total_duration:.2f} seconds")
        print(f"Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens} | Total Tokens: {total_tokens}")
        
        tokens_per_second = 0
        if completion_tokens > 0 and total_duration > 0:
            tokens_per_second = completion_tokens / total_duration
            print(f"Tokens per Second (T/s): {tokens_per_second:.2f}")
        else:
            print("Tokens per Second (T/s): N/A")

        # --- Response Quality Check ---
        full_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        repetition_ratio = 1.0 # Default to no repetition
        lines = full_content.split('\n')
        if len(lines) > 10:
            unique_lines = len(set(l for l in lines if l.strip()))
            total_lines = len([l for l in lines if l.strip()])
            if total_lines > 0:
                repetition_ratio = unique_lines / total_lines
                if repetition_ratio < 0.4:
                    print("⚠️  WARNING: High repetition detected in response")
        
        if full_content.count('```') % 2 != 0:
            print("⚠️  WARNING: Unclosed code block detected")
            
        # Save Response
        filename_safe_name = name.lower().replace(":", "").replace(" ", "_")
        output_path = os.path.join("test_results", f"{filename_safe_name}_response.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_content)
                
        print(f"\n--- 📜 Response Preview ---")
        print(textwrap.shorten(full_content, width=400, placeholder="..."))
        print(f"\n--- Full response saved to '{output_path}' ---")

        # --- Performance Summary ---
        with open(os.path.join("test_results", "performance_summary.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n{name}\n")
            f.write(f"Duration: {total_duration:.2f}s | ")
            f.write(f"Tokens: {completion_tokens} | ") # Changed to completion tokens for T/s consistency
            if tokens_per_second > 0:
                f.write(f"T/s: {tokens_per_second:.2f}")
                if repetition_ratio < 0.4:
                    f.write(" [HIGH REPETITION]")
            else:
                f.write(f"T/s: N/A")
            f.write("\n")
            f.write("-" * 80)

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
    print("--- Advanced Linux Agent Test Suite (100 Complex Challenges) ---")
    print(f"Targeting Server: {API_URL}")
    print(f"Total Tests: {len(TEST_CASES)}")
    
    summary_file = os.path.join("test_results", "performance_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=== ADVANCED REASONING TEST SUITE - PERFORMANCE SUMMARY ===\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80)
    
    successful_tests = 0
    failed_tests = 0
    
    start_time_total = time.time()
    for i, test in enumerate(TEST_CASES):
        print(f"\n\n[{i+1}/{len(TEST_CASES)}] Starting test...")
        if run_performance_test(test):
            successful_tests += 1
        else:
            failed_tests += 1
        
        time.sleep(1) # Small delay between tests
    
    end_time_total = time.time()
    total_suite_duration = end_time_total - start_time_total

    # Final Summary
    print(f"\n{'='*80}")
    print(f"--- 📈 FINAL SUMMARY ---")
    print(f"Total Suite Duration: {total_suite_duration:.2f} seconds")
    print(f"Total Tests: {len(TEST_CASES)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    success_rate = (successful_tests / len(TEST_CASES)) * 100 if len(TEST_CASES) > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"\n--- ✅ Test suite completed. Check '{summary_file}' for summary. ---")
    print(f"{'='*80}")
    
    # Append summary to file
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"\n\nFINAL SUMMARY\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Suite Duration: {total_suite_duration:.2f} seconds\n")
        f.write(f"Total Tests: {len(TEST_CASES)}\n")
        f.write(f"Successful: {successful_tests}\n")
        f.write(f"Failed: {failed_tests}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n")