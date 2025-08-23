import requests
import json
import time
import os
import textwrap
import re
from collections import Counter

# Configuration for base model completions
API_URL = "http://127.0.0.1:8001/v1/completions"  # Completions endpoint, not chat
HEADERS = {
    "Content-Type": "application/json"
}
os.makedirs("test_results", exist_ok=True)

TEST_CASES = [
    # ========== TEXT CONTINUATION TESTS (1-10) ==========
    {
        "name": "Test 1: Natural Text Continuation",
        "prompt": "The discovery of penicillin by Alexander Fleming in 1928 marked a turning point in medical history. This accidental discovery occurred when Fleming noticed that a mold had contaminated one of his bacterial cultures, creating a clear zone where bacteria couldn't grow. The implications of this observation were",
        "params": {
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False,
            "stop": ["\n\n", "###"]
        }
    },
    {
        "name": "Test 2: Technical Documentation Continuation",
        "prompt": "## API Documentation: RESTful User Management\n\n### Overview\nThis API provides endpoints for managing user accounts in a multi-tenant application.\n\n### Authentication\nAll endpoints require Bearer token authentication in the header:\n```\nAuthorization: Bearer <token>\n```\n\n### Endpoints\n\n#### GET /api/v1/users\nRetrieves a paginated list of users.\n\n**Query Parameters:**\n- `page` (integer): Page number, default 1\n- `limit` (integer): Items per page, default 20\n- `sort` (string): Sort field, options include",
        "params": {
            "max_tokens": 300,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 3: Narrative Story Continuation",
        "prompt": "The old lighthouse keeper had seen many storms in his forty years of service, but nothing quite like this one. The waves crashed against the rocks with unprecedented fury, and the wind howled like a creature possessed. As he climbed the spiral staircase to check the light, he noticed something peculiar through the window - a ship that seemed to be sailing directly into the storm rather than away from it. The vessel's lights flickered in an unusual pattern, almost as if",
        "params": {
            "max_tokens": 400,
            "temperature": 0.8,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 4: Scientific Abstract Completion",
        "prompt": "Abstract: Recent advances in quantum computing have demonstrated the potential for exponential speedup in certain computational tasks. In this paper, we present a novel quantum algorithm for solving the traveling salesman problem (TSP) that achieves a quadratic speedup over classical approaches. Our method combines variational quantum eigensolvers with",
        "params": {
            "max_tokens": 250,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 5: Code Function Completion",
        "prompt": "def merge_sort(arr):\n    \"\"\"\n    Implements the merge sort algorithm to sort an array in ascending order.\n    Time complexity: O(n log n)\n    Space complexity: O(n)\n    \"\"\"\n    if len(arr) <= 1:\n        return arr\n    \n    # Divide the array into two halves\n    mid = len(arr) // 2\n    left_half = arr[:mid]\n    right_half = arr[mid:]\n    \n    # Recursively sort both halves\n    left_sorted = merge_sort(left_half)\n    right_sorted = merge_sort(right_half)\n    \n    # Merge the sorted halves\n    merged = []\n    i = j = 0\n    \n    while i < len(left_sorted) and j < len(right_sorted):",
        "params": {
            "max_tokens": 300,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 6: News Article Continuation",
        "prompt": "WASHINGTON (Reuters) - The Federal Reserve announced today a quarter-point increase in interest rates, marking the third consecutive rate hike this year. The decision, which was widely anticipated by market analysts, comes as inflation remains above the central bank's 2% target despite recent signs of cooling in some sectors.\n\nFed Chair Jerome Powell stated in a press conference that",
        "params": {
            "max_tokens": 250,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 7: Poetry Continuation",
        "prompt": "In the quiet hours before dawn breaks,\nWhen shadows dance and silence wakes,\nThe world holds its breath in patient wait,\nFor morning's light to seal its fate.\n\nThe stars above begin to fade,\nAs night's dark curtain starts to trade\nIts velvet cloak for golden streams,",
        "params": {
            "max_tokens": 150,
            "temperature": 0.9,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 8: Dialogue Continuation",
        "prompt": "\"I don't understand,\" Sarah said, staring at the ancient map spread across the table. \"These symbols don't match any known language.\"\n\nDr. Martinez adjusted his glasses and leaned closer. \"That's because they predate written language as we know it. Look at the pattern here,\" he traced his finger along a series of intricate marks. \"These aren't words, they're",
        "params": {
            "max_tokens": 200,
            "temperature": 0.75,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 9: Recipe Continuation",
        "prompt": "Classic French Onion Soup Recipe\n\nIngredients:\n- 6 large yellow onions, thinly sliced\n- 4 tablespoons butter\n- 2 tablespoons olive oil\n- 1 teaspoon sugar\n- 8 cups beef broth\n- 1/2 cup dry white wine\n- Salt and pepper to taste\n- French baguette, sliced\n- 2 cups grated Gruyère cheese\n\nInstructions:\n1. In a large pot, melt butter with olive oil over medium heat.\n2. Add onions and sugar, stirring to coat evenly.\n3. Cook onions for 40-45 minutes, stirring occasionally, until they are deep golden brown and caramelized.\n4.",
        "params": {
            "max_tokens": 250,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 10: Email Continuation",
        "prompt": "Subject: Q3 Financial Report - Key Highlights and Action Items\n\nDear Team,\n\nI hope this email finds you well. I wanted to share some important updates from our Q3 financial report that was finalized yesterday.\n\nKey Highlights:\n• Revenue increased by 12% year-over-year, exceeding our projections by $2.3M\n• Operating expenses were reduced by 8% through our efficiency initiatives\n• Customer acquisition cost decreased by 15% while maintaining quality\n\nHowever, there are some areas that require immediate attention:",
        "params": {
            "max_tokens": 200,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": False
        }
    },

    # ========== PATTERN COMPLETION TESTS (11-20) ==========
    {
        "name": "Test 11: Numerical Pattern Completion",
        "prompt": "The Fibonacci sequence is one of the most famous patterns in mathematics:\n1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765,",
        "params": {
            "max_tokens": 100,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 12: List Pattern Completion",
        "prompt": "Common HTTP Status Codes:\n200 - OK\n201 - Created\n204 - No Content\n301 - Moved Permanently\n302 - Found\n304 - Not Modified\n400 - Bad Request\n401 - Unauthorized\n403 - Forbidden\n404 - Not Found\n405 - Method Not Allowed\n409 - Conflict\n500 - Internal Server Error\n502 - Bad Gateway\n503 -",
        "params": {
            "max_tokens": 150,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 13: Markdown Table Completion",
        "prompt": "| Programming Language | Year Created | Creator | Paradigm |\n|---------------------|--------------|---------|----------|\n| C | 1972 | Dennis Ritchie | Procedural |\n| Python | 1991 | Guido van Rossum | Multi-paradigm |\n| Java | 1995 | James Gosling | Object-oriented |\n| JavaScript | 1995 | Brendan Eich | Multi-paradigm |\n| Go | 2009 | Google | Concurrent |\n| Rust | 2010 | Mozilla | Systems |\n| Swift | 2014 |",
        "params": {
            "max_tokens": 200,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 14: JSON Structure Completion",
        "prompt": "{\n  \"users\": [\n    {\n      \"id\": 1,\n      \"name\": \"Alice Johnson\",\n      \"email\": \"alice@example.com\",\n      \"role\": \"admin\",\n      \"created_at\": \"2024-01-15T10:30:00Z\"\n    },\n    {\n      \"id\": 2,\n      \"name\": \"Bob Smith\",\n      \"email\": \"bob@example.com\",\n      \"role\": \"user\",\n      \"created_at\": \"2024-02-20T14:45:00Z\"\n    },\n    {\n      \"id\": 3,",
        "params": {
            "max_tokens": 200,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 15: SQL Query Completion",
        "prompt": "-- Complex query to find top customers by purchase value\nSELECT \n    c.customer_id,\n    c.first_name,\n    c.last_name,\n    c.email,\n    COUNT(DISTINCT o.order_id) as total_orders,\n    SUM(oi.quantity * oi.unit_price) as total_spent,\n    AVG(oi.quantity * oi.unit_price) as avg_order_value,\n    MAX(o.order_date) as last_order_date\nFROM customers c\nINNER JOIN orders o ON c.customer_id = o.customer_id\nINNER JOIN order_items oi ON o.order_id = oi.order_id\nWHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)\nGROUP BY",
        "params": {
            "max_tokens": 150,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 16: CSS Style Completion",
        "prompt": ".responsive-card {\n    display: flex;\n    flex-direction: column;\n    padding: 1.5rem;\n    margin: 1rem;\n    border-radius: 8px;\n    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);\n    transition: transform 0.3s ease, box-shadow 0.3s ease;\n    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\n}\n\n.responsive-card:hover {\n    transform:",
        "params": {
            "max_tokens": 150,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 17: Mathematical Proof Continuation",
        "prompt": "Theorem: The square root of 2 is irrational.\n\nProof by contradiction:\nAssume √2 is rational. Then √2 = p/q where p and q are integers with no common factors (the fraction is in lowest terms).\n\nSquaring both sides: 2 = p²/q²\nMultiplying both sides by q²: 2q² = p²\n\nThis means p² is even, which implies p must be even (since the square of an odd number is odd).\nLet p = 2k for some integer k.\n\nSubstituting: 2q² = (2k)² = 4k²\nSimplifying: q² = 2k²\n\nThis means q² is even, which implies",
        "params": {
            "max_tokens": 200,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 18: Changelog Entry Completion",
        "prompt": "# Changelog\n\n## [2.3.0] - 2024-08-15\n### Added\n- New dark mode theme with automatic OS detection\n- Export functionality for PDF and CSV formats\n- Keyboard shortcuts for common actions (Ctrl+S, Ctrl+N, Ctrl+O)\n\n### Changed\n- Improved performance of data grid rendering by 40%\n- Updated dependencies to latest stable versions\n- Redesigned settings page for better UX\n\n### Fixed\n- Memory leak in real-time data updates\n- Incorrect timestamp formatting in exported files\n- Crash when handling malformed JSON responses\n\n## [2.2.1] - 2024-07-20\n### Fixed\n-",
        "params": {
            "max_tokens": 200,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 19: Unit Test Completion",
        "prompt": "import unittest\nfrom datetime import datetime, timedelta\nfrom user_service import UserService, User, UserNotFoundError\n\nclass TestUserService(unittest.TestCase):\n    def setUp(self):\n        self.service = UserService()\n        self.test_user = User(\n            id=1,\n            username=\"testuser\",\n            email=\"test@example.com\",\n            created_at=datetime.now()\n        )\n    \n    def test_create_user_success(self):\n        user = self.service.create_user(\"newuser\", \"new@example.com\")\n        self.assertIsNotNone(user.id)\n        self.assertEqual(user.username, \"newuser\")\n        self.assertEqual(user.email, \"new@example.com\")\n    \n    def test_get_user_by_id(self):",
        "params": {
            "max_tokens": 250,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 20: Regular Expression Pattern Completion",
        "prompt": "Common Regular Expression Patterns:\n\nEmail validation: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$\nPhone number (US): ^\\+?1?\\s?\\(?\\d{3}\\)?[\\s.-]?\\d{3}[\\s.-]?\\d{4}$\nURL validation: ^(https?|ftp)://[^\\s/$.?#].[^\\s]*$\nIPv4 address: ^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$\nDate (YYYY-MM-DD):",
        "params": {
            "max_tokens": 150,
            "temperature": 0.2,
            "top_p": 0.95,
            "stream": False
        }
    },

    # ========== STYLE MIMICRY TESTS (21-30) ==========
    {
        "name": "Test 21: Academic Writing Style",
        "prompt": "The proliferation of artificial intelligence in contemporary society has engendered multifaceted debates regarding its ethical implications. Scholars have articulated divergent perspectives on the autonomy of intelligent systems, with particular emphasis on the epistemological challenges inherent in defining consciousness within computational frameworks. Furthermore, the intersection of machine learning algorithms with societal infrastructure necessitates",
        "params": {
            "max_tokens": 200,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 22: Casual Blog Style",
        "prompt": "Hey everyone! 🚀 So I've been tinkering with this new productivity app for the past few weeks, and honestly? It's been a total game-changer. You know how we all struggle with the whole 'getting things done' thing, right? Well, this app basically",
        "params": {
            "max_tokens": 200,
            "temperature": 0.8,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 23: Legal Document Style",
        "prompt": "WHEREAS, the Party of the First Part (hereinafter referred to as \"Licensor\") is the sole owner of certain intellectual property rights, including but not limited to patents, trademarks, and copyrights pertaining to the software described in Exhibit A; and\n\nWHEREAS, the Party of the Second Part (hereinafter referred to as \"Licensee\") desires to obtain a non-exclusive license to use, distribute, and modify said software subject to the terms and conditions set forth herein;\n\nNOW, THEREFORE, in consideration of the mutual covenants and agreements contained herein, and for other good and valuable consideration, the receipt and sufficiency of which are hereby acknowledged,",
        "params": {
            "max_tokens": 200,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 24: Children's Story Style",
        "prompt": "Once upon a time, in a magical forest where the trees whispered secrets and the flowers danced in the moonlight, there lived a little rabbit named Benny. Benny had the fluffiest white tail and the biggest, brightest eyes you ever did see! Every morning, Benny would hop-hop-hop through the meadow, looking for adventures. One sunny day, while munching on sweet clover, Benny discovered",
        "params": {
            "max_tokens": 200,
            "temperature": 0.85,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 25: Technical Manual Style",
        "prompt": "4.3.2 Calibration Procedure\n\nWARNING: Ensure power supply is disconnected before beginning calibration.\n\nStep 1: Remove the access panel by unscrewing the four Phillips head screws (M4x12mm) located at corners A1, A2, B1, and B2 (refer to Figure 4.3).\n\nStep 2: Locate the calibration potentiometer (R12) on the main PCB. Using a non-conductive adjustment tool, rotate the potentiometer clockwise until the voltage reading on test point TP3 measures exactly 2.500V ±0.005V.\n\nStep 3:",
        "params": {
            "max_tokens": 200,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 26: Victorian Literature Style",
        "prompt": "It was a peculiarly dreary autumn evening when Mr. Thornbury first made the acquaintance of Lady Weatherstone at the manor house upon the moor. The drawing room, illuminated by the flickering gaslight, cast long shadows upon the Persian carpets, whilst the rain beat incessantly against the mullioned windows. Lady Weatherstone, resplendent in her emerald velvet gown, extended a gloved hand with such grace that",
        "params": {
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 27: Sports Commentary Style",
        "prompt": "AND HERE WE GO, FOLKS! Williams takes the inbound pass, dribbles up court with just 12 seconds left on the clock. The crowd is on their feet! He crosses over at half court, Thompson's right on him - WHAT A MOVE! Williams spins past the defender, drives to the lane, the defense collapses and",
        "params": {
            "max_tokens": 150,
            "temperature": 0.75,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 28: Medical Report Style",
        "prompt": "PATIENT HISTORY:\nThe patient is a 45-year-old male presenting with complaints of intermittent chest pain radiating to the left arm, onset 3 days ago. Pain is described as 6/10 in severity, pressure-like in quality, exacerbated by physical exertion and partially relieved by rest. Associated symptoms include mild dyspnea and diaphoresis. No reported nausea, vomiting, or syncope.\n\nPAST MEDICAL HISTORY:\nHypertension (diagnosed 2019)\nType 2 Diabetes Mellitus (diagnosed 2021)\nHyperlipidemia (diagnosed 2020)\n\nPHYSICAL EXAMINATION:",
        "params": {
            "max_tokens": 200,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 29: Movie Script Style",
        "prompt": "INT. ABANDONED WAREHOUSE - NIGHT\n\nThe door CREAKS open. SARAH (30s, determined, bloodstained jacket) enters cautiously, gun drawn. Moonlight streams through broken windows, casting eerie shadows.\n\nSARAH\n(whispering into radio)\nI'm in position. No sign of--\n\nA NOISE from the shadows. Sarah spins, aiming her weapon.\n\nVOICE (O.S.)\n(calm, menacing)\nI've been expecting you, Detective.\n\nSarah's eyes narrow. She recognizes the voice.\n\nSARAH\n(steady, but tense)\n",
        "params": {
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 30: Product Review Style",
        "prompt": "⭐⭐⭐⭐ (4/5 stars)\n\nPROS:\n✓ Excellent build quality - feels premium and sturdy\n✓ Battery life is impressive (lasted 3 days with moderate use)\n✓ Setup was surprisingly easy, took less than 10 minutes\n✓ The app integration works flawlessly\n\nCONS:\n✗ Price point is a bit high compared to competitors\n✗ Limited color options (only black and white available)\n✗ The instruction manual could be clearer\n\nOVERALL THOUGHTS:\nI've been using this product for about a month now, and",
        "params": {
            "max_tokens": 200,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": False
        }
    },

    # ========== CONTEXT COHERENCE TESTS (31-40) ==========
    {
        "name": "Test 31: Long Context Maintenance",
        "prompt": "In 1953, a young scientist named Dr. Elena Vasquez discovered an unusual mineral formation in the caves beneath the Andes Mountains. The mineral, which she initially catalogued as EV-1953, exhibited properties that defied conventional understanding of crystalline structures. Under specific lighting conditions, the crystals appeared to shift between different geometric configurations, a phenomenon she documented meticulously in her field notes.\n\nTwenty years later, in 1973, her former student, Dr. James Chen, revisited her research while working on semiconductor materials. He noticed that the transformation patterns Vasquez had observed in EV-1953 closely resembled the quantum tunneling effects he was seeing in his silicon experiments. This connection led him to propose a radical hypothesis: that Vasquez had inadvertently discovered",
        "params": {
            "max_tokens": 300,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 32: Character Consistency",
        "prompt": "Marcus had always been meticulous about three things: his appearance, his schedule, and his coffee. Every morning at precisely 6:47 AM, he would grind exactly 18 grams of Ethiopian Yirgacheffe beans, heat water to 96 degrees Celsius, and brew for exactly 4 minutes. His colleagues at the architectural firm often joked that you could set your watch by Marcus's routines.\n\nThis particular Tuesday started differently. Marcus woke at 6:51 AM - four minutes late. The delay cascaded through his morning routine like dominoes. By the time he reached the office at 8:23 AM instead of his usual 8:15 AM, his assistant Jennifer immediately noticed something was wrong. Marcus's tie was slightly askew, his usual perfect part in his hair was absent, and most shocking of all,",
        "params": {
            "max_tokens": 250,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 33: Technical Specification Consistency",
        "prompt": "System Architecture Overview:\n\nThe DataFlow 3000 processing unit consists of four main components:\n1. Input Buffer (IB-4K): 4KB FIFO buffer with 120MHz clock speed\n2. Processing Core (PC-X7): Dual-pipeline architecture, 3.2 GHz base frequency\n3. Cache Memory (CM-L2): 2MB L2 cache with 8-way set associative mapping\n4. Output Controller (OC-DMA): Direct memory access with 64-bit addressing\n\nThe IB-4K receives data packets at a maximum rate of 10 Gbps through the ethernet interface. Each packet is validated using CRC-32 checksums before being passed to the PC-X7. The processing core can handle up to 1000 operations per clock cycle, utilizing both pipelines simultaneously.\n\nGiven these specifications, the theoretical maximum throughput of the system when processing 512-byte packets would be",
        "params": {
            "max_tokens": 200,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 34: Historical Timeline Consistency",
        "prompt": "The Maravian Empire Timeline:\n\n1247 CE: King Aldric III establishes the Maravian Empire after uniting the seven northern kingdoms.\n1263 CE: The Great Library of Maravia is founded, becoming the largest repository of knowledge in the known world.\n1289 CE: Prince Casimir, son of Aldric III, leads the first expedition across the Eastern Desert.\n1294 CE: The Silk Road extension reaches Maravia, bringing unprecedented wealth through trade.\n1301 CE: Queen Isabella (Casimir's daughter) ascends to the throne at age 16.\n1308 CE: The Maravian Navy is established with 200 warships.\n1315 CE: The Golden Age begins with the discovery of silver mines in the southern provinces.\n1323 CE: Queen Isabella commissions the Grand Cathedral, which would take 50 years to complete.\n\nIn 1330 CE, during the seventh year of the cathedral's construction,",
        "params": {
            "max_tokens": 250,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 35: Scientific Method Description",
        "prompt": "Experimental Protocol for Photosynthesis Rate Measurement:\n\nObjective: Determine the effect of light intensity on the rate of photosynthesis in Elodea canadensis.\n\nMaterials:\n- 5 specimens of Elodea canadensis (10cm each)\n- 500mL beakers (5)\n- LED light sources (100W, 75W, 50W, 25W, 0W control)\n- Sodium bicarbonate solution (0.2% concentration)\n- Digital thermometer\n- Oxygen sensor probe\n- Data logger\n\nProcedure:\n1. Fill each beaker with 400mL of sodium bicarbonate solution\n2. Place one Elodea specimen in each beaker\n3. Position light sources 30cm from each beaker\n4. Allow 10 minutes for acclimatization\n5. Begin measuring oxygen production at 1-minute intervals for 30 minutes\n6.",
        "params": {
            "max_tokens": 200,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 36: Business Case Analysis",
        "prompt": "Executive Summary: Digital Transformation Initiative\n\nCurrent State:\nOur company currently operates with legacy systems averaging 15 years old. Manual processes account for 60% of operational workflows. Customer data is siloed across 12 different databases with no real-time synchronization. Average customer service response time is 48 hours.\n\nProposed Solution:\nImplement a cloud-based ERP system with integrated CRM capabilities. Estimated investment: $3.2 million over 18 months.\n\nExpected Benefits:\n- Reduce operational costs by 35% within 2 years\n- Improve customer response time to under 4 hours\n- Eliminate data silos through unified database architecture\n- Enable real-time analytics for decision making\n\nRisk Assessment:\nThe primary risks include resistance to change from employees accustomed to current systems, potential data migration issues, and temporary productivity decline during the transition period. To mitigate these risks, we propose",
        "params": {
            "max_tokens": 250,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 37: Geographic Description",
        "prompt": "The Patagonian region spans approximately 1,000,000 square kilometers across southern Argentina and Chile. Bordered by the Atlantic Ocean to the east and the Pacific Ocean to the west, this vast territory is characterized by dramatic geographical diversity. The Andes Mountains form its western spine, with peaks reaching over 3,000 meters, while the eastern portions consist of vast steppes and plateaus.\n\nThe climate varies significantly from west to east. The western slopes receive abundant rainfall, sometimes exceeding 4,000mm annually, creating temperate rainforests. In contrast, the eastern steppes lie in a rain shadow, receiving less than 200mm of precipitation yearly. This stark difference in precipitation has created",
        "params": {
            "max_tokens": 200,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 38: Recipe Method Continuation",
        "prompt": "Sourdough Bread - Day 3 of Process\n\nYour starter should now be active and doubling in size within 4-6 hours of feeding. Today we begin the actual bread making.\n\nMorning (8:00 AM):\n1. In a large bowl, mix 100g active starter with 375g lukewarm water until dissolved\n2. Add 500g bread flour and mix until just combined (no dry flour visible)\n3. Cover and let rest for 30 minutes (autolyse phase)\n\n8:30 AM:\n4. Add 10g salt and 25g water, mixing by hand until fully incorporated\n5. Perform the first set of stretch and folds: grab one side of the dough, stretch up and fold over to the opposite side. Rotate bowl 90 degrees and repeat 4 times total.\n\n9:00 AM - 11:00 AM:\n6. Perform stretch and folds every 30 minutes (total of 4 sets)\n7. The dough should become smoother and more elastic with each set\n\n11:00 AM - 2:00 PM:\n8.",
        "params": {
            "max_tokens": 250,
            "temperature": 0.5,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 39: Mystery Plot Development",
        "prompt": "Detective Morrison examined the crime scene methodically. The victim, a renowned art dealer named Vincent Graves, lay sprawled on the gallery floor surrounded by priceless paintings. The peculiar thing was that nothing appeared to be stolen, yet the security system had been expertly disabled between 2:17 and 2:43 AM.\n\nThree details caught Morrison's attention:\n1. A single white rose placed deliberately on the victim's chest\n2. The number \"7\" drawn in dust on a nearby display case\n3. All the paintings had been rotated exactly 15 degrees clockwise\n\nThe security footage from neighboring buildings showed four people entering the gallery district that night:\n- A woman in a red coat at 1:45 AM\n- A delivery driver at 2:05 AM\n- Two men in business suits at 2:20 AM\n- A jogger at 2:55 AM\n\nMorrison's phone buzzed. The forensics team had found",
        "params": {
            "max_tokens": 250,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False
        }
    },
    {
        "name": "Test 40: Tutorial Instruction Flow",
        "prompt": "Building Your First React Component - Step by Step\n\nPrerequisites:\n- Node.js installed (version 14 or higher)\n- Basic knowledge of JavaScript and HTML\n- A code editor (VS Code recommended)\n\nStep 1: Create a new React app\nOpen your terminal and run:\n```bash\nnpx create-react-app my-first-component\ncd my-first-component\n```\n\nStep 2: Create a component file\nIn the src folder, create a new file called 'Welcome.js'\n\nStep 3: Write the component\n```javascript\nimport React from 'react';\n\nfunction Welcome({ name, age }) {\n  return (\n    <div className=\"welcome-container\">\n      <h1>Hello, {name}!</h1>\n      <p>You are {age} years old.</p>\n    </div>\n  );\n}\n\nexport default Welcome;\n```\n\nStep 4: Import and use the component\nOpen src/App.js and",
        "params": {
            "max_tokens": 250,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": False
        }
    }
]

def calculate_perplexity_approximation(text):
    """
    Calculate a simple approximation of perplexity based on vocabulary diversity
    and sentence structure complexity.
    """
    words = text.lower().split()
    unique_words = set(words)
    
    # Vocabulary diversity ratio
    diversity = len(unique_words) / len(words) if words else 0
    
    # Approximate perplexity score (lower is better)
    # This is a simplified metric for demonstration
    perplexity_score = 1 / (diversity + 0.01) if diversity > 0 else float('inf')
    
    return perplexity_score

def analyze_completion_quality(prompt, completion):
    """
    Analyze the quality of a completion for a base model.
    """
    metrics = {}
    
    # Check if completion continues naturally from prompt
    prompt_last_sentence = prompt.strip().split('.')[-1]
    completion_first_sentence = completion.strip().split('.')[0]
    
    # Coherence check - does it flow naturally?
    metrics['starts_naturally'] = not completion.strip().startswith(('However', 'But', 'Although')) or '.' in prompt[-5:]
    
    # Length check
    metrics['word_count'] = len(completion.split())
    
    # Repetition check
    sentences = completion.split('.')
    unique_sentences = set(sentences)
    metrics['repetition_ratio'] = len(unique_sentences) / len(sentences) if sentences else 0
    
    # Check for common base model issues
    metrics['has_markdown'] = '```' in completion
    metrics['has_lists'] = any(line.strip().startswith(('- ', '* ', '1.', '2.')) for line in completion.split('\n'))
    
    # Vocabulary diversity
    words = completion.lower().split()
    unique_words = set(words)
    metrics['vocabulary_diversity'] = len(unique_words) / len(words) if words else 0
    
    return metrics

def run_performance_test(test_case):
    """Sends a request to the base model API and analyzes performance."""
    
    name = test_case["name"]
    prompt = test_case["prompt"]
    params = test_case["params"]
    
    print(f"\n{'='*80}\n--- 🚀 RUNNING: {name} ---\n{'='*80}")
    
    # Print prompt preview
    print("--- Prompt Preview ---")
    print(textwrap.shorten(prompt, width=300, placeholder="..."))
    print("-" * 40)

    # Prepare payload for completion endpoint (not chat)
    payload = {
        "model": "gpt-oss-20b",  # Specify model name
        "prompt": prompt,  # Direct prompt, no messages array
        **params
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        response.raise_for_status()
        end_time = time.time()

        response_data = response.json()
        
        print("\n--- ✅ SUCCESS: Server Responded ---")
        
        # Performance Metrics
        total_duration = end_time - start_time
        
        # Handle different response structures
        if "usage" in response_data:
            usage_data = response_data["usage"]
            completion_tokens = usage_data.get("completion_tokens", 0)
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            total_tokens = usage_data.get("total_tokens", 0)
        else:
            # Estimate tokens if not provided
            completion_text = response_data.get("choices", [{}])[0].get("text", "")
            completion_tokens = len(completion_text.split()) * 1.3  # Rough estimate
            prompt_tokens = len(prompt.split()) * 1.3
            total_tokens = completion_tokens + prompt_tokens
        
        print(f"\n--- 📊 Performance Metrics ---")
        print(f"Total Request Time: {total_duration:.2f} seconds")
        print(f"Prompt Tokens: {int(prompt_tokens)} | Completion Tokens: {int(completion_tokens)} | Total Tokens: {int(total_tokens)}")
        
        tokens_per_second = 0
        if completion_tokens > 0 and total_duration > 0:
            tokens_per_second = completion_tokens / total_duration
            print(f"Tokens per Second (T/s): {tokens_per_second:.2f}")
        
        # Get completion text
        completion_text = response_data.get("choices", [{}])[0].get("text", "")
        
        # Calculate perplexity approximation
        perplexity = calculate_perplexity_approximation(completion_text)
        print(f"Perplexity (approximate): {perplexity:.2f}")
        
        # Analyze completion quality
        quality_metrics = analyze_completion_quality(prompt, completion_text)
        
        print(f"\n--- 📈 Quality Metrics ---")
        print(f"Word Count: {quality_metrics['word_count']}")
        print(f"Vocabulary Diversity: {quality_metrics['vocabulary_diversity']:.2%}")
        print(f"Flows Naturally: {'Yes' if quality_metrics['starts_naturally'] else 'No'}")
        print(f"Repetition Ratio: {quality_metrics['repetition_ratio']:.2%}")
        
        # Save Response
        filename_safe_name = name.lower().replace(":", "").replace(" ", "_")
        output_path = os.path.join("test_results", f"{filename_safe_name}_completion.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"PROMPT:\n{prompt}\n\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"COMPLETION:\n{completion_text}\n\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"METRICS:\n")
            f.write(f"Duration: {total_duration:.2f}s\n")
            f.write(f"Tokens/sec: {tokens_per_second:.2f}\n")
            f.write(f"Perplexity: {perplexity:.2f}\n")
            for key, value in quality_metrics.items():
                f.write(f"{key}: {value}\n")
                
        print(f"\n--- 📜 Completion Preview ---")
        print(textwrap.shorten(completion_text, width=400, placeholder="..."))
        print(f"\n--- Full completion saved to '{output_path}' ---")

        # Performance Summary
        with open(os.path.join("test_results", "gpt_oss_20b_summary.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n{name}\n")
            f.write(f"Duration: {total_duration:.2f}s | ")
            f.write(f"Tokens: {int(completion_tokens)} | ")
            f.write(f"T/s: {tokens_per_second:.2f} | ")
            f.write(f"Perplexity: {perplexity:.2f} | ")
            f.write(f"Quality: {quality_metrics['vocabulary_diversity']:.2%} diversity\n")
            f.write("-" * 80)

        return True

    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: Request timed out for '{name}'")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: Request failed for '{name}'")
        print(f"   Details: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during '{name}': {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("GPT-OSS-20B BASE MODEL TEST SUITE")
    print("Testing completion, continuation, and generation capabilities")
    print("=" * 80)
    print(f"Server: {API_URL}")
    print(f"Total Tests: {len(TEST_CASES)}")
    print("=" * 80)
    
    # Initialize summary file
    summary_file = os.path.join("test_results", "gpt_oss_20b_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("GPT-OSS-20B BASE MODEL TEST RESULTS\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80)
    
    successful_tests = 0
    failed_tests = 0
    
    start_time_total = time.time()
    
    for i, test in enumerate(TEST_CASES):
        print(f"\n\n[{i+1}/{len(TEST_CASES)}] Preparing test...")
        if run_performance_test(test):
            successful_tests += 1
        else:
            failed_tests += 1
        
        # Small delay between tests
        time.sleep(0.5)
    
    end_time_total = time.time()
    total_suite_duration = end_time_total - start_time_total

    # Final Summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total Duration: {total_suite_duration:.2f} seconds")
    print(f"Successful Tests: {successful_tests}/{len(TEST_CASES)}")
    print(f"Failed Tests: {failed_tests}/{len(TEST_CASES)}")
    print(f"Success Rate: {(successful_tests/len(TEST_CASES)*100):.1f}%")
    print(f"Average Time per Test: {total_suite_duration/len(TEST_CASES):.2f} seconds")
    
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"\n\nFINAL SUMMARY\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {total_suite_duration:.2f} seconds\n")
        f.write(f"Success Rate: {(successful_tests/len(TEST_CASES)*100):.1f}%\n")