# ======================================================================================
# LONG-CONTEXT BENCHMARK TEST SUITE
# ======================================================================================
# This suite contains 10 challenging benchmark scenarios designed to rigorously test 
# a base model's performance on tasks requiring deep understanding of ~32,000 token 
# context windows. Each scenario follows the exact format: {"name", "prompt", "validator"}
# ======================================================================================

LONG_CONTEXT_BENCHMARKS = [
    # ======================================================================================
    # CATEGORY 1: MULTI-DOCUMENT SYNTHESIS (3 scenarios)
    # ======================================================================================
    
    {
        "name": "Financial Report Synthesis and Investment Recommendation",
        "prompt": """
CONTEXT: You are a senior financial analyst tasked with providing investment recommendations based on comprehensive analysis of multiple financial reports. Below are complete annual reports for three competing companies in the renewable energy sector.

REPORT 1: SOLARTEC INDUSTRIES ANNUAL REPORT 2023
===============================================
EXECUTIVE SUMMARY
SolarTec Industries (STI) completed fiscal year 2023 with record revenues of $4.2 billion, representing a 28% increase from the previous year. Our market-leading solar panel technology, combined with strategic acquisitions in energy storage, positioned us as the dominant force in the residential solar market. Key achievements include the successful launch of our PowerMax 500W panels, which achieved 22.5% efficiency ratings, and the acquisition of three battery manufacturing facilities across Texas, Nevada, and Arizona.

FINANCIAL HIGHLIGHTS:
- Total Revenue: $4.2B (+28% YoY)
- Gross Profit: $1.47B (35% margin)
- Operating Income: $588M (14% margin)
- Net Income: $378M (9% margin)  
- Cash and Cash Equivalents: $892M
- Total Debt: $1.2B
- R&D Investment: $294M (7% of revenue)
- Capital Expenditures: $420M

BUSINESS SEGMENTS PERFORMANCE:
Residential Solar Division: $2.8B revenue (+35% YoY), driven by PowerMax panel adoption and expanded dealer network reaching 2,400 certified installers nationwide. Average selling price per installation increased 12% to $18,500 while maintaining 34% gross margins.

Commercial Solar Division: $940M revenue (+18% YoY), benefiting from new federal tax incentives and long-term contracts with Fortune 500 companies. Notable wins include 50MW installation for Walmart distribution centers and 75MW project for Amazon fulfillment facilities.

Energy Storage Division: $460M revenue (+45% YoY), primarily from our newly acquired battery operations. GridStore home battery systems captured 18% market share in residential sector. Commercial-scale storage projects generated $180M in revenue.

OPERATIONAL METRICS:
- Manufacturing Capacity: 2.4 GW annually across 8 facilities
- Panels Shipped: 1.8 million units (950MW total)
- Installation Network: 2,400 certified dealers
- Employee Count: 18,400 (+2,100 YoY)
- Customer Satisfaction: 94% (industry-leading)
- Safety Record: 0.12 incidents per 100,000 hours worked

RESEARCH & DEVELOPMENT:
Our $294M R&D investment focused on next-generation perovskite-silicon tandem cells, targeting 30% efficiency by 2025. Breakthrough achievements include the development of weather-resistant coatings extending panel life to 35 years and smart inverter technology reducing installation costs by 15%. Filed 127 patents in 2023, bringing total patent portfolio to 890 active patents.

MARKET POSITION & COMPETITION:
SolarTec maintains #1 position in US residential market with 28% share, followed by SunPower (18%) and Enphase (15%). Our vertical integration strategy provides cost advantages and supply chain resilience. Independent studies confirm our PowerMax panels achieve highest performance-to-cost ratio in premium segment.

SUSTAINABILITY INITIATIVES:
Achieved carbon neutrality across all operations through renewable energy adoption and carbon offset programs. Launched panel recycling program processing 15,000 units in pilot phase. Water usage reduced 22% through closed-loop manufacturing processes. Obtained B-Corp certification recognizing social and environmental performance.

RISKS & CHALLENGES:
Regulatory uncertainty regarding federal tax credit extensions remains primary concern. Chinese manufacturing competition pressures margins despite tariff protections. Raw material costs, particularly polysilicon and silver, experienced 18% inflation. Skilled technician shortage constrains installation capacity growth. Grid interconnection delays average 6-8 months in key markets.

OUTLOOK & GUIDANCE:
Management projects 2024 revenue growth of 22-25% to $5.1-5.3B, driven by continued residential market expansion and commercial pipeline of $1.8B in contracted projects. Gross margins expected to improve to 37-38% through manufacturing automation and PowerMax 2.0 launch. Planned investments include $600M in new manufacturing capacity and $350M in R&D focusing on energy management software platforms.

CEO LETTER TO SHAREHOLDERS:
"2023 marked a transformational year for SolarTec as we solidified our position as America's solar leader. Our strategic focus on innovation, vertical integration, and customer experience delivered exceptional results while positioning us for sustained growth. The renewable energy transition is accelerating, driven by economic competitiveness, climate urgency, and supportive policies. SolarTec is uniquely positioned to capitalize on this megatrend through our technology leadership, operational excellence, and financial strength. We remain committed to our mission of making clean energy accessible to every American household while delivering superior returns to our shareholders."

FINANCIAL STATEMENTS DETAIL:
Income Statement (in millions):
Revenue: $4,200
Cost of Goods Sold: $2,730
Gross Profit: $1,470
Operating Expenses:
  - Sales & Marketing: $504
  - General & Administrative: $210  
  - Research & Development: $294
  - Depreciation: $168
Total Operating Expenses: $1,176
Operating Income: $588
Interest Expense: $42
Other Income: $18
Income Before Taxes: $564
Tax Expense: $186
Net Income: $378

Balance Sheet (in millions):
Assets:
Current Assets:
  - Cash & Equivalents: $892
  - Accounts Receivable: $315
  - Inventory: $420
  - Prepaid Expenses: $28
  Total Current Assets: $1,655
Property, Plant & Equipment (net): $1,890
Intangible Assets: $245
Other Assets: $95
Total Assets: $3,885

Liabilities & Equity:
Current Liabilities:
  - Accounts Payable: $280
  - Accrued Expenses: $195
  - Short-term Debt: $150
  Total Current Liabilities: $625
Long-term Debt: $1,050
Other Liabilities: $125
Total Liabilities: $1,800
Shareholders' Equity: $2,085
Total Liabilities & Equity: $3,885

Cash Flow Statement (in millions):
Operating Cash Flow: $567
Investing Cash Flow: $(485)
  - Capital Expenditures: $(420)
  - Acquisitions: $(85)
  - Other: $20
Financing Cash Flow: $(45)
  - Debt Repayment: $(120)
  - Equity Issuance: $85
  - Dividends: $(10)
Net Change in Cash: $37

QUARTERLY BREAKDOWN:
Q1 2023: Revenue $920M, EPS $0.18
Q2 2023: Revenue $1,050M, EPS $0.22  
Q3 2023: Revenue $1,180M, EPS $0.28
Q4 2023: Revenue $1,050M, EPS $0.25

REPORT 2: WINDFORCE ENERGY CORPORATION ANNUAL REPORT 2023
========================================================
CHAIRMAN'S MESSAGE
WindForce Energy Corporation delivered solid performance in 2023 despite challenging market conditions including supply chain disruptions and regulatory headwinds. Our diversified portfolio spanning onshore wind, offshore wind, and grid infrastructure services generated $3.8 billion in revenue while maintaining industry-leading safety standards. The completion of our 500MW Atlantic Wind project off the Massachusetts coast represents a milestone achievement, establishing WindForce as a premier offshore wind developer.

FINANCIAL PERFORMANCE OVERVIEW:
- Total Revenue: $3.8B (+12% YoY)
- Gross Profit: $1.14B (30% margin)
- Operating Income: $342M (9% margin)
- Net Income: $198M (5.2% margin)
- Cash Position: $524M
- Total Debt: $2.1B
- R&D Spending: $152M (4% of revenue)
- Capital Investments: $680M

SEGMENT ANALYSIS:
Onshore Wind Development: $2.1B revenue (+8% YoY), with 850MW of new capacity commissioned across 12 wind farms in Texas, Oklahoma, and Iowa. Average turbine capacity factor of 38% exceeded industry benchmark of 35%. Long-term power purchase agreements (PPAs) secured at average price of $32/MWh provide stable cash flows for 20-year terms.

Offshore Wind Development: $780M revenue (+35% YoY), driven by Atlantic Wind project completion and Massachusetts state renewable energy mandates. Pipeline includes 2.2GW across 4 projects in federal lease areas. Supply chain partnerships with Vestas and GE secured turbine delivery through 2027.

Grid Services & Transmission: $920M revenue (+15% YoY), providing interconnection and transmission solutions for wind projects. Expansion into energy storage integration services captured growing demand for grid stability solutions. Notable projects include 200MW battery installation in California supporting grid reliability.

OPERATIONAL EXCELLENCE:
- Total Installed Capacity: 3.2 GW across 45 wind farms
- Power Generation: 8.4 TWh annually
- Capacity Factor: 37% average (industry-leading)
- Workforce: 8,900 employees across 15 states
- Safety Performance: 0.89 TRIR (total recordable incident rate)
- Environmental Impact: Avoided 4.2M tons CO2 annually

TECHNOLOGY & INNOVATION:
Our $152M R&D investment advanced next-generation turbine technology optimized for low-wind conditions, increasing capacity factors by 8% compared to previous generation. Digital twin modeling and predictive maintenance systems reduced unplanned downtime 25%. Offshore floating platform design enables development in deeper waters previously inaccessible.

MARKET DYNAMICS:
US wind industry installed 10.3 GW in 2023, with WindForce capturing 8.2% market share. Production tax credit extensions through 2025 provide policy certainty supporting project economics. Transmission constraints in key markets limit growth potential, with average interconnection queue time extending to 3.5 years.

COMPETITIVE LANDSCAPE:
NextEra Energy leads market with 25% share, followed by Berkshire Hathaway Energy (18%) and Avangrid (12%). WindForce ranks #4 nationally with growing offshore specialization. Differentiation through advanced technology, project execution capabilities, and strategic partnerships with equipment manufacturers.

SUSTAINABILITY COMMITMENTS:
Achieved net-zero operational emissions through renewable energy sourcing and efficiency improvements. Launched turbine blade recycling program addressing end-of-life concerns. Biodiversity protection measures implemented across all projects including bird and bat monitoring systems. Community investment programs contributed $25M to rural economic development.

RISK FACTORS:
Policy uncertainty regarding federal tax incentives creates project financing challenges. Transmission infrastructure limitations constrain development in high-wind resource areas. Supply chain disruptions and commodity price inflation pressured project margins. Permitting delays average 18 months for new projects. Opposition from local communities requires extensive stakeholder engagement.

2024 STRATEGIC PRIORITIES:
Advancing 2.2GW offshore pipeline through permitting and financing processes. Expanding grid services capabilities through strategic acquisitions. Developing floating offshore technology for California market entry. Optimizing operations through digital transformation initiatives. Strengthening balance sheet through asset monetization and refinancing activities.

FINANCIAL STATEMENTS DETAILS:
Revenue Composition:
- Onshore Wind: $2,100M (55%)
- Offshore Wind: $780M (21%)  
- Grid Services: $920M (24%)

Geographic Revenue:
- Texas: $1,140M (30%)
- Northeast: $912M (24%)
- Midwest: $836M (22%)
- Other: $912M (24%)

Cost Structure:
- Direct Costs: $2,660M (70% of revenue)
- SG&A: $456M (12% of revenue)
- Depreciation: $342M (9% of revenue)
- Interest Expense: $126M

Asset Base:
- Wind Farms (net): $4,200M
- Equipment & Infrastructure: $1,680M
- Cash & Working Capital: $720M
- Goodwill & Intangibles: $340M
Total Assets: $6,940M

Debt Profile:
- Term Loans: $1,260M
- Project Finance: $630M
- Senior Notes: $210M
- Total Debt: $2,100M
- Weighted Average Interest Rate: 6.2%
- Average Maturity: 12 years

QUARTERLY PERFORMANCE:
Q1: Revenue $890M, EBITDA $185M
Q2: Revenue $970M, EBITDA $205M
Q3: Revenue $1,020M, EBITDA $225M
Q4: Revenue $920M, EBITDA $195M

REPORT 3: HYDROGEN SOLUTIONS INC. ANNUAL REPORT 2023
===================================================
CEO MESSAGE TO STAKEHOLDERS
HydroGen Solutions delivered breakthrough performance in 2023, establishing our leadership position in the emerging hydrogen economy. Revenue of $1.9 billion exceeded guidance by 15%, driven by accelerating demand for clean hydrogen solutions across industrial, transportation, and power generation applications. Our proprietary electrolysis technology achieved industry-leading efficiency ratings while strategic partnerships positioned us for exponential growth as hydrogen adoption scales globally.

CONSOLIDATED FINANCIAL RESULTS:
- Total Revenue: $1.9B (+85% YoY)
- Gross Profit: $665M (35% margin)
- Operating Income: $152M (8% margin)
- Net Income: $95M (5% margin)
- Cash & Short-term Investments: $430M
- Total Debt: $650M
- R&D Investment: $247M (13% of revenue)
- Capital Expenditures: $285M

BUSINESS UNIT PERFORMANCE:
Electrolysis Equipment Manufacturing: $1.1B revenue (+95% YoY), benefiting from 200% capacity expansion and next-generation HydroMax electrolyzer launch. Unit costs reduced 30% through manufacturing automation and supply chain optimization. Order backlog reached $2.4B providing visibility through 2025.

Hydrogen Production & Distribution: $540M revenue (+75% YoY), operating 15 production facilities with combined capacity of 120 tons daily. Green hydrogen production costs achieved $4.50/kg, approaching cost parity with grey hydrogen. Long-term offtake agreements secured with steel, ammonia, and transportation customers.

Engineering & Construction Services: $260M revenue (+65% YoY), designing and building large-scale hydrogen production facilities for industrial customers. Notable projects include 50MW electrolysis plant for European steel producer and 100MW facility for ammonia manufacturer. Project margins improved to 18% through standardized designs.

TECHNOLOGY LEADERSHIP:
Our $247M R&D investment delivered breakthrough advances in proton exchange membrane (PEM) electrolysis efficiency, achieving 75% system efficiency compared to industry average of 70%. Stack lifetime extended to 80,000 hours through advanced materials and design optimization. Patent portfolio expanded to 240 active patents covering electrolysis, storage, and fuel cell technologies.

MARKET OPPORTUNITY:
Global hydrogen market projected to reach $200B by 2030, driven by decarbonization mandates and cost competitiveness. Transportation sector adoption accelerating with 15,000 fuel cell vehicles deployed in California and expanding infrastructure networks. Industrial applications in steel, chemicals, and refining represent $50B near-term opportunity.

STRATEGIC PARTNERSHIPS:
Formed joint venture with Toyota for fuel cell vehicle applications, combining HydroGen's production capabilities with Toyota's automotive expertise. Strategic partnership with Air Liquide provides European market access and distribution network. Collaboration with Shell on hydrogen fueling infrastructure development across North America.

MANUFACTURING & OPERATIONS:
- Production Facilities: 8 manufacturing sites globally
- Electrolyzer Production Capacity: 2.5 GW annually
- Hydrogen Production Capacity: 120 tons daily
- Employee Base: 4,200 professionals worldwide
- Quality Certification: ISO 9001, AS9100 aerospace
- Safety Performance: Zero lost-time incidents in 2023

REGULATORY ENVIRONMENT:
Inflation Reduction Act provides $3/kg production tax credit for clean hydrogen, improving project economics significantly. California's hydrogen hub funding award of $1.2B supports infrastructure development. European Union's REPowerEU plan targets 10 million tons annual production by 2030, creating substantial export opportunities.

ENVIRONMENTAL IMPACT:
Green hydrogen production avoided 800,000 tons CO2 emissions compared to conventional grey hydrogen. Lifecycle assessment confirms 90% emissions reduction versus fossil fuel alternatives. Water consumption optimized through closed-loop systems and renewable energy integration.

COMPETITIVE POSITIONING:
Market leaders include ITM Power (15% share), Nel ASA (12%), and Plug Power (10%). HydroGen's 8% market share growing rapidly through technology differentiation and manufacturing scale. Focus on large-scale industrial applications provides margin premium over commodity products.

FINANCIAL OUTLOOK:
Management projects 2024 revenue growth of 60-70% to $3.0-3.2B, supported by $2.4B order backlog and expanding production capacity. Gross margins expected to improve to 38-40% through operational leverage and pricing optimization. Planned investments include $400M facility expansion and $300M R&D acceleration.

RISK CONSIDERATIONS:
Regulatory changes affecting hydrogen incentives could impact project economics. Technology competition from alkaline and solid oxide electrolysis alternatives. Raw material cost inflation, particularly platinum group metals. Skilled workforce shortage in emerging hydrogen sector. Customer concentration risk with top 5 customers representing 45% of revenue.

DETAILED FINANCIAL ANALYSIS:
Revenue Growth Drivers:
- Equipment Sales: +95% driven by capacity expansion
- Production Services: +75% from new facility commissioning
- Engineering Services: +65% from project pipeline growth

Geographic Distribution:
- North America: $1,140M (60%)
- Europe: $570M (30%)
- Asia-Pacific: $190M (10%)

Cost Management:
Manufacturing costs reduced 25% through automation and scale effects. SG&A expenses maintained at 15% of revenue despite growth. R&D intensity at 13% supports technology leadership position.

Balance Sheet Strength:
Working capital increased to $285M supporting revenue growth. Property, plant & equipment expanded to $1,200M through capacity investments. Debt-to-equity ratio maintained at conservative 0.45x.

Cash Flow Generation:
- Operating Cash Flow: $285M (+120% YoY)
- Free Cash Flow: $145M after capital investments
- Cash Conversion Cycle: 65 days, improving from 78 days
- Return on Invested Capital: 12.5%

FORWARD-LOOKING STATEMENTS:
Multiple expansion opportunities in hydrogen value chain including storage, transportation, and fuel cells. Potential acquisitions to accelerate technology development and market penetration. International expansion focused on Europe and Asia-Pacific markets with local partnerships.

After analyzing these three comprehensive annual reports, provide a detailed investment recommendation that addresses the following specific questions:

1. Which company demonstrates the strongest financial performance and why? Consider revenue growth, profitability margins, cash generation, and balance sheet strength.

2. What are the key competitive advantages and risks for each company? Analyze technology differentiation, market positioning, regulatory exposure, and operational capabilities.

3. Which renewable energy technology (solar, wind, or hydrogen) presents the most attractive investment opportunity over the next 5 years? Consider market size, growth potential, policy support, and technological maturity.

4. Based on your analysis, rank the three companies from most attractive to least attractive investment opportunity. Provide specific financial metrics and strategic rationale supporting your recommendation.

5. What portfolio allocation would you recommend across these three companies for a $10 million renewable energy investment fund? Explain your diversification strategy and risk management approach.

Your analysis must demonstrate synthesis of the financial data, strategic positioning, and market dynamics presented in all three reports to reach evidence-based investment conclusions.""",
        "validator": "Response must provide a complete investment recommendation addressing all 5 specific questions with quantitative analysis and ranking of SolarTec, WindForce, and HydroGen based on financial metrics from the reports."
    },
    
    {
        "name": "Medical Research Paper Synthesis for Treatment Protocol",
        "prompt": """
CONTEXT: You are a medical researcher tasked with developing evidence-based treatment recommendations by synthesizing findings from multiple clinical studies. Below are three complete research papers on novel treatments for Type 2 diabetes.

STUDY 1: GLUCAGON-LIKE PEPTIDE-1 RECEPTOR AGONIST EFFICACY IN TYPE 2 DIABETES
Journal: New England Journal of Medicine
Authors: Dr. Sarah Chen, MD, PhD et al., Stanford University Medical Center
Study Type: Randomized, Double-Blind, Placebo-Controlled Trial
Duration: 52 weeks
Participants: 1,247 adults with Type 2 diabetes

ABSTRACT:
Background: Glucagon-like peptide-1 (GLP-1) receptor agonists represent a promising therapeutic approach for Type 2 diabetes management. This study evaluated the efficacy and safety of weekly semaglutide compared to placebo in patients inadequately controlled on metformin monotherapy.

Methods: We randomly assigned 1,247 participants (mean age 58.3 years, 52% female, mean HbA1c 8.4%) to receive either semaglutide 1.0 mg weekly subcutaneous injection or matching placebo for 52 weeks. Primary endpoint was change in HbA1c from baseline. Secondary endpoints included weight loss, blood pressure reduction, and cardiovascular risk markers.

Results: At 52 weeks, semaglutide group demonstrated significantly greater HbA1c reduction (-1.8% vs -0.3% placebo, p<0.001). Body weight decreased by 6.8 kg in semaglutide group versus 1.2 kg in placebo group (p<0.001). Systolic blood pressure reduced by 7.2 mmHg versus 2.1 mmHg placebo (p<0.001). Cardiovascular events occurred in 2.4% semaglutide patients versus 4.1% placebo patients (HR 0.58, 95% CI 0.35-0.95, p=0.031).

Adverse Events: Gastrointestinal side effects were most common in semaglutide group, with nausea (42% vs 8% placebo), vomiting (18% vs 3%), and diarrhea (25% vs 7%). Serious adverse events occurred in 8.2% semaglutide versus 11.4% placebo patients. Hypoglycemic episodes were rare (3.1% vs 2.8% placebo).

DETAILED METHODOLOGY:
Inclusion Criteria: Adults aged 18-75 with Type 2 diabetes duration ≥6 months, HbA1c 7.0-10.5%, BMI 25-45 kg/m², stable metformin therapy ≥3 months at dose ≥1500 mg daily or maximum tolerated dose.

Exclusion Criteria: Type 1 diabetes, insulin therapy within 90 days, history of pancreatitis, severe renal impairment (eGFR <30 mL/min/1.73m²), active cardiovascular disease within 6 months, pregnancy or nursing.

Randomization: Centralized computerized randomization stratified by baseline HbA1c (<8.5% vs ≥8.5%) and BMI (<30 vs ≥30 kg/m²). Treatment allocation concealed using sealed envelope system.

Study Drug Administration: Semaglutide 1.0 mg administered via pre-filled pen injector subcutaneously once weekly. Matching placebo injections contained identical excipients without active drug. Injection technique training provided at baseline.

Efficacy Assessments: HbA1c measured using standardized HPLC method at central laboratory. Fasting plasma glucose assessed using glucose oxidase method. Body weight measured using calibrated scales at each visit. Blood pressure recorded as mean of three seated measurements after 5-minute rest.

Safety Monitoring: Adverse events recorded and coded using MedDRA terminology. Laboratory safety assessments included complete blood count, comprehensive metabolic panel, liver function tests, lipase, and urinalysis at weeks 0, 12, 24, 36, and 52.

Statistical Analysis: Primary analysis used modified intention-to-treat population including all randomized patients receiving ≥1 dose of study medication. Missing data handled using mixed-effects model for repeated measures. Superiority testing used two-sided alpha level of 0.05.

BASELINE CHARACTERISTICS:
Semaglutide Group (n=623):
- Age: 58.1 ± 10.7 years
- Female: 324 (52%)
- Race: White 412 (66%), Hispanic 124 (20%), Black 87 (14%)
- Duration of diabetes: 7.2 ± 4.8 years
- HbA1c: 8.39 ± 0.92%
- Fasting glucose: 168 ± 42 mg/dL
- Body weight: 89.3 ± 18.4 kg
- BMI: 32.1 ± 5.2 kg/m²
- Systolic BP: 134 ± 16 mmHg
- eGFR: 82.4 ± 18.7 mL/min/1.73m²

Placebo Group (n=624):
- Age: 58.5 ± 11.2 years
- Female: 327 (52%)
- Race: White 418 (67%), Hispanic 121 (19%), Black 85 (14%)
- Duration of diabetes: 7.4 ± 5.1 years
- HbA1c: 8.42 ± 0.89%
- Fasting glucose: 171 ± 45 mg/dL
- Body weight: 90.1 ± 19.2 kg
- BMI: 32.3 ± 5.4 kg/m²
- Systolic BP: 135 ± 17 mmHg
- eGFR: 81.8 ± 19.3 mL/min/1.73m²

EFFICACY RESULTS TIMELINE:
Week 12:
- HbA1c change: -1.2% semaglutide vs -0.1% placebo
- Weight change: -4.1 kg semaglutide vs -0.8 kg placebo
- FPG change: -32 mg/dL semaglutide vs -5 mg/dL placebo

Week 24:
- HbA1c change: -1.5% semaglutide vs -0.2% placebo
- Weight change: -5.8 kg semaglutide vs -1.0 kg placebo
- FPG change: -41 mg/dL semaglutide vs -7 mg/dL placebo

Week 52 (Primary Endpoint):
- HbA1c change: -1.8% semaglutide vs -0.3% placebo (p<0.001)
- Weight change: -6.8 kg semaglutide vs -1.2 kg placebo (p<0.001)
- FPG change: -47 mg/dL semaglutide vs -9 mg/dL placebo (p<0.001)
- Systolic BP change: -7.2 mmHg vs -2.1 mmHg placebo (p<0.001)

TARGET ACHIEVEMENT:
HbA1c <7.0%: 68.3% semaglutide vs 18.2% placebo patients
Weight loss ≥5%: 72.1% semaglutide vs 22.4% placebo patients
Weight loss ≥10%: 31.8% semaglutide vs 4.2% placebo patients

CARDIOVASCULAR OUTCOMES:
Major Adverse Cardiovascular Events (MACE):
- Cardiovascular death: 0.3% semaglutide vs 0.8% placebo
- Non-fatal myocardial infarction: 0.8% semaglutide vs 1.4% placebo
- Non-fatal stroke: 0.6% semaglutide vs 1.1% placebo
- Hospitalization for unstable angina: 0.7% semaglutide vs 0.8% placebo
- Composite MACE: 2.4% semaglutide vs 4.1% placebo (HR 0.58, p=0.031)

STUDY 2: SODIUM-GLUCOSE COTRANSPORTER-2 INHIBITOR CARDIOVASCULAR OUTCOMES
Journal: The Lancet
Authors: Dr. Michael Rodriguez, MD et al., Mayo Clinic
Study Type: Randomized, Double-Blind, Placebo-Controlled Cardiovascular Outcome Trial
Duration: Median 4.2 years
Participants: 17,160 patients with Type 2 diabetes and established cardiovascular disease

ABSTRACT:
Background: Sodium-glucose cotransporter-2 (SGLT2) inhibitors improve glycemic control and reduce body weight in Type 2 diabetes. This cardiovascular outcome trial evaluated the effect of empagliflozin on major adverse cardiovascular events in high-risk patients.

Methods: We randomized 17,160 patients with Type 2 diabetes and established cardiovascular disease to empagliflozin 10 mg daily, empagliflozin 25 mg daily, or placebo. Primary composite endpoint was cardiovascular death, non-fatal myocardial infarction, or non-fatal stroke. Secondary endpoints included hospitalization for heart failure and all-cause mortality.

Results: During median follow-up of 4.2 years, primary endpoint occurred in 10.5% empagliflozin patients versus 12.1% placebo patients (HR 0.86, 95% CI 0.74-0.99, p=0.038). Cardiovascular death reduced by 38% (HR 0.62, 95% CI 0.49-0.77, p<0.001). Heart failure hospitalization reduced by 35% (HR 0.65, 95% CI 0.50-0.85, p=0.002). All-cause mortality reduced by 32% (HR 0.68, 95% CI 0.57-0.82, p<0.001).

Glycemic Effects: HbA1c reduced by 0.54% versus placebo at 1 year, with sustained reduction throughout study. Body weight decreased by 2.2 kg versus placebo. Systolic blood pressure reduced by 4.16 mmHg versus placebo.

Safety Profile: Genital infections increased in empagliflozin group (6.4% vs 1.8% placebo). Urinary tract infections were similar between groups. Diabetic ketoacidosis was rare (0.1% empagliflozin vs <0.1% placebo). Volume depletion events occurred in 2.3% empagliflozin versus 1.4% placebo patients.

COMPREHENSIVE METHODOLOGY:
Study Population: Patients recruited from 590 sites across 42 countries. Inclusion required Type 2 diabetes with HbA1c ≤10.0% and established cardiovascular disease defined as history of myocardial infarction, multi-vessel coronary artery disease, single-vessel coronary disease with symptoms, or stroke >2 months prior to screening.

Randomization Strategy: 1:1:1 allocation to empagliflozin 10 mg, empagliflozin 25 mg, or placebo using interactive voice/web response system. Stratification by HbA1c (<8.5% vs ≥8.5%), BMI (<30 vs ≥30 kg/m²), and geographic region.

Background Therapy: Standard diabetes medications continued throughout study. Glucose-lowering therapy intensification permitted per investigator discretion following standardized algorithm. Cardiovascular medications optimized per guidelines.

Endpoint Adjudication: Independent clinical events committee blinded to treatment assignment adjudicated all potential endpoints using pre-specified criteria. Source documents reviewed for all suspected events.

Statistical Considerations: Event-driven design required 691 primary endpoint events for 90% power to detect 20% relative risk reduction. Interim analyses performed by independent data monitoring committee.

BASELINE DEMOGRAPHICS:
Total Population (N=17,160):
- Age: 63.1 ± 8.6 years
- Female: 5,677 (33.1%)
- Race: White 13,728 (80.0%), Asian 1,716 (10.0%), Black 686 (4.0%)
- Diabetes duration: 16.9 ± 8.4 years
- HbA1c: 8.07 ± 0.85%
- eGFR: 74.3 ± 21.4 mL/min/1.73m²
- BMI: 30.6 ± 5.3 kg/m²

Cardiovascular History:
- Previous myocardial infarction: 13,416 (78.2%)
- Multi-vessel coronary disease: 8,409 (49.0%)
- Previous stroke: 4,634 (27.0%)
- Peripheral artery disease: 4,803 (28.0%)
- Heart failure: 1,716 (10.0%)

Medications at Baseline:
- ACE inhibitors/ARBs: 14,414 (84.0%)
- Beta-blockers: 13,502 (78.7%)
- Statins: 14,757 (86.0%)
- Antiplatelet therapy: 15,001 (87.4%)
- Metformin: 12,012 (70.0%)
- Insulin: 8,236 (48.0%)
- Sulfonylureas: 7,550 (44.0%)

PRIMARY OUTCOME RESULTS:
Empagliflozin pooled: 772/11,440 patients (10.5%)
Placebo: 690/5,720 patients (12.1%)
Hazard ratio: 0.86 (95% CI 0.74-0.99)
P-value: 0.038
Number needed to treat: 63 patients over 4.2 years

SECONDARY OUTCOME RESULTS:
Cardiovascular Death:
- Empagliflozin: 3.7% vs Placebo: 5.9%
- HR 0.62 (95% CI 0.49-0.77), p<0.001

Heart Failure Hospitalization:
- Empagliflozin: 2.7% vs Placebo: 4.1%
- HR 0.65 (95% CI 0.50-0.85), p=0.002

All-Cause Mortality:
- Empagliflozin: 5.7% vs Placebo: 8.3%
- HR 0.68 (95% CI 0.57-0.82), p<0.001

Non-fatal Myocardial Infarction:
- Empagliflozin: 4.8% vs Placebo: 5.4%
- HR 0.87 (95% CI 0.70-1.09), p=0.226

Non-fatal Stroke:
- Empagliflozin: 3.5% vs Placebo: 3.0%
- HR 1.18 (95% CI 0.89-1.56), p=0.257

STUDY 3: DUAL GLUCOSE-DEPENDENT INSULINOTROPIC POLYPEPTIDE/GLP-1 RECEPTOR AGONIST
Journal: Diabetes Care
Authors: Dr. Jennifer Kim, MD, PhD et al., University of Pennsylvania
Study Type: Phase 3, Randomized, Open-Label, Active-Controlled Trial
Duration: 40 weeks
Participants: 1,879 patients with Type 2 diabetes inadequately controlled on metformin

ABSTRACT:
Background: Dual glucose-dependent insulinotropic polypeptide (GIP) and glucagon-like peptide-1 (GLP-1) receptor agonism represents a novel therapeutic approach for Type 2 diabetes. This study compared the efficacy and safety of tirzepatide versus semaglutide in patients inadequately controlled on metformin.

Methods: We randomized 1,879 patients to tirzepatide 5 mg, 10 mg, or 15 mg weekly subcutaneous injection versus semaglutide 1 mg weekly for 40 weeks. Primary endpoint was non-inferiority of tirzepatide versus semaglutide for HbA1c reduction at 40 weeks. Key secondary endpoint was superiority testing for weight reduction.

Results: All tirzepatide doses demonstrated non-inferiority and superiority to semaglutide for HbA1c reduction. Tirzepatide 15 mg achieved -2.4% HbA1c reduction versus -1.9% with semaglutide (p<0.001). Weight loss was significantly greater with all tirzepatide doses: 15 mg dose achieved -12.4 kg versus -6.2 kg with semaglutide (p<0.001).

Tolerability: Gastrointestinal adverse events were dose-related in tirzepatide groups but similar to semaglutide. Nausea occurred in 12-22% of tirzepatide patients versus 20% with semaglutide. Treatment discontinuation rates were 4.3-7.1% across tirzepatide doses versus 7.2% with semaglutide.

DETAILED STUDY DESIGN:
Participants: Adults aged 18-75 years with Type 2 diabetes for ≥3 months, HbA1c 7.0-10.5%, BMI ≥25 kg/m², stable metformin therapy ≥3 months. Exclusion criteria included insulin therapy within 3 months, history of pancreatitis, and severe renal impairment.

Treatment Groups:
- Tirzepatide 5 mg weekly (n=470)
- Tirzepatide 10 mg weekly (n=469)  
- Tirzepatide 15 mg weekly (n=470)
- Semaglutide 1 mg weekly (n=470)

Dose Escalation: Tirzepatide initiated at 2.5 mg weekly, escalated every 4 weeks to target dose. Semaglutide initiated at 0.25 mg weekly, escalated to 1 mg over 16 weeks per label.

BASELINE CHARACTERISTICS:
Mean across all groups:
- Age: 57.4 ± 10.1 years
- Female: 52.7%
- Race: White 69.2%, Hispanic 16.8%, Black 9.1%, Asian 4.9%
- Diabetes duration: 8.6 ± 6.2 years
- HbA1c: 8.28 ± 0.97%
- Body weight: 86.4 ± 19.8 kg
- BMI: 31.9 ± 6.0 kg/m²
- eGFR: 89.7 ± 17.2 mL/min/1.73m²

EFFICACY OUTCOMES AT 40 WEEKS:

HbA1c Reduction (Least Squares Mean):
- Tirzepatide 5 mg: -2.07% (95% CI -2.19 to -1.94)
- Tirzepatide 10 mg: -2.24% (95% CI -2.37 to -2.12)
- Tirzepatide 15 mg: -2.37% (95% CI -2.49 to -2.25)
- Semaglutide 1 mg: -1.86% (95% CI -1.98 to -1.73)

Treatment Difference vs Semaglutide:
- 5 mg: -0.21% (p=0.002)
- 10 mg: -0.38% (p<0.001)  
- 15 mg: -0.51% (p<0.001)

Body Weight Reduction:
- Tirzepatide 5 mg: -7.6 kg (-8.7%)
- Tirzepatide 10 mg: -10.3 kg (-11.9%)
- Tirzepatide 15 mg: -12.4 kg (-14.7%)
- Semaglutide 1 mg: -6.2 kg (-7.1%)

HbA1c Target Achievement:
<7.0%:
- Tirzepatide 5 mg: 87.4%
- Tirzepatide 10 mg: 90.9%
- Tirzepatide 15 mg: 92.6%
- Semaglutide: 78.7%

≤6.5%:
- Tirzepatide 5 mg: 73.2%
- Tirzepatide 10 mg: 81.0%
- Tirzepatide 15 mg: 86.4%
- Semaglutide: 58.5%

Weight Loss Categories:
≥5% weight loss:
- Tirzepatide 5 mg: 85.2%
- Tirzepatide 10 mg: 89.1%
- Tirzepatide 15 mg: 91.5%
- Semaglutide: 78.9%

≥15% weight loss:
- Tirzepatide 5 mg: 27.7%
- Tirzepatide 10 mg: 42.6%
- Tirzepatide 15 mg: 57.4%
- Semaglutide: 19.1%

SAFETY AND TOLERABILITY:

Adverse Events Leading to Discontinuation:
- Tirzepatide 5 mg: 4.3%
- Tirzepatide 10 mg: 5.1%
- Tirzepatide 15 mg: 7.1%
- Semaglutide: 7.2%

Gastrointestinal Adverse Events:
Nausea:
- Tirzepatide 5 mg: 12.2%
- Tirzepatide 10 mg: 16.8%
- Tirzepatide 15 mg: 22.1%
- Semaglutide: 19.6%

Diarrhea:
- Tirzepatide 5 mg: 12.7%
- Tirzepatide 10 mg: 16.4%
- Tirzepatide 15 mg: 18.3%
- Semaglutide: 15.7%

Vomiting:
- Tirzepatide 5 mg: 3.4%
- Tirzepatide 10 mg: 6.4%
- Tirzepatide 15 mg: 8.9%
- Semaglutide: 8.1%

Based on comprehensive analysis of these three clinical studies, develop evidence-based treatment recommendations that address:

1. Which medication class demonstrates superior glycemic control efficacy and by what mechanisms? Compare HbA1c reduction, target achievement rates, and durability of effect across the three studies.

2. What are the cardiovascular safety and efficacy profiles of each treatment approach? Analyze MACE reduction, heart failure outcomes, and mortality benefits based on the available data.

3. How do weight loss effects compare between treatments and what clinical implications does this have for patient selection? Consider magnitude of weight reduction and patient subgroups most likely to benefit.

4. What are the key safety considerations and contraindications for each medication class? Evaluate gastrointestinal tolerability, hypoglycemia risk, and serious adverse events.

5. Develop a clinical decision algorithm for treatment selection based on patient characteristics, comorbidities, and treatment goals. Consider factors such as cardiovascular risk, weight management needs, and tolerability concerns.

Your recommendations must integrate the quantitative data from all three studies to provide evidence-based guidance for optimizing Type 2 diabetes treatment outcomes.""",
        "validator": "Response must provide evidence-based treatment recommendations addressing all 5 specific questions with quantitative comparisons of semaglutide, empagliflozin, and tirzepatide based on efficacy, safety, and cardiovascular outcomes data from the studies."
    },
    
    {
        "name": "Climate Change Policy Analysis from Multiple Government Reports",
        "prompt": """
CONTEXT: You are a senior policy analyst tasked with synthesizing climate policy recommendations from multiple comprehensive government reports. Below are three complete government climate assessment documents that must be analyzed together to develop unified policy recommendations.

REPORT 1: NATIONAL CLIMATE ASSESSMENT - FOURTH NATIONAL CLIMATE ASSESSMENT VOLUME II
U.S. Global Change Research Program
Executive Summary and Key Findings

EXECUTIVE SUMMARY
Climate change is transforming where and how we live and presenting growing challenges to human health and quality of life, the economy, and the natural systems that support us. This assessment concludes that the evidence of human-caused climate change is overwhelming and continues to strengthen, that the impacts of climate change are intensifying across the country, and that climate-related threats to Americans' physical, social, and economic well-being are rising.

KEY FINDINGS:

1. COMMUNITIES: Climate change affects human health, livelihood, infrastructure, and social systems across the nation. Rural areas face economic challenges as climate impacts agricultural productivity and infrastructure. Urban areas face threats from heat, flooding, and air quality degradation affecting millions of Americans.

Economic Impacts: Annual losses from extreme weather events exceed $100 billion annually. Heat-related deaths could increase by thousands annually by 2090. Coastal property damage from sea level rise projected at $14.2 billion annually by 2045. Agricultural productivity losses estimated at 10-25% by 2050 without adaptation.

Health Impacts: Temperature-related deaths projected to increase 27,000 annually by 2100 under high emissions scenario. Air quality degradation causing additional 57,000 premature deaths annually by 2050. Vector-borne disease expansion affecting millions in previously unaffected regions.

Infrastructure Vulnerabilities: 95% of coastal infrastructure at risk from storm surge and sea level rise. Transportation systems face $5.1 billion annual costs from extreme weather. Power grid reliability declining due to increased cooling demand and storm frequency.

2. ECONOMY: Without substantial and sustained global mitigation and regional adaptation efforts, climate change is expected to cause growing losses to American infrastructure and property and impede the rate of economic growth over this century.

Sector-Specific Economic Analysis:
Agriculture: Crop yields declining in key regions. Corn belt productivity falling 16% by 2050. Heat stress reducing livestock productivity 25% by 2080. Irrigation demand increasing 24% while water availability decreasing 12%.

Energy: Cooling degree days increasing 50-90% by 2050, raising electricity demand $30 billion annually. Power plant efficiency declining due to higher temperatures. Renewable energy potential increasing in some regions while decreasing in others.

Tourism: Ski industry facing 50% reduction in season length by 2050. Beach tourism threatened by erosion and extreme weather. National parks experiencing visitation pattern shifts costing $40 billion in economic impact.

Coastal Economy: $1 trillion in coastal property at risk by 2100. Fishing industry facing species migration and stock collapse. Port operations disrupted by extreme weather causing supply chain impacts.

3. WATER: Rising temperatures reduce snowpack and increase evaporation, increasing drought risks and water supply vulnerabilities. Extreme precipitation events cause flooding that damages infrastructure and contaminates water supplies.

Regional Water Analysis:
Southwest: Colorado River flow declining 20% by 2050. Aquifer depletion accelerating due to increased irrigation demand. Municipal water supplies stressed in Phoenix, Las Vegas, and Los Angeles metropolitan areas.

Southeast: Hurricane-driven flooding contaminating freshwater supplies. Saltwater intrusion affecting coastal aquifers. Drought frequency doubling by 2050 impacting agricultural regions.

Great Lakes: Water level fluctuations affecting shipping and coastal communities. Increased algal blooms due to warming temperatures and changing precipitation patterns.

4. HEALTH: Impacts within the United States include increases in extreme weather events, changes in air and water quality, potential disruptions to food systems, and threats to human health and safety.

Temperature-Related Health Impacts: Heat wave frequency tripling by 2050. Vulnerable populations including elderly, children, and outdoor workers face disproportionate risks. Urban heat island effects amplifying exposure in cities.

Air Quality: Ozone formation increasing with higher temperatures. Wildfire smoke affecting air quality across larger geographic areas. Allergen production extending seasonal duration and intensity.

Vector-Borne Diseases: Tick-borne disease expansion into northern regions. Mosquito-borne illness potential increasing with longer warm seasons and changing precipitation patterns.

Food Safety: Pathogen growth rates increasing with temperature. Foodborne illness incidence projected to increase 10% per degree of warming.

5. INDIGENOUS PEOPLES: Many indigenous communities are on the front lines of climate change and face significant risks to their health, well-being, cultural traditions, and economic livelihoods from climate-related changes to ecosystems and environments.

Cultural and Traditional Impacts: Subsistence hunting and fishing patterns disrupted by species migration and ecosystem changes. Traditional foods becoming less available or safe to consume. Sacred sites threatened by sea level rise and extreme weather.

Economic Impacts: Tourism revenue declining in climate-dependent communities. Traditional industries like fishing and agriculture facing productivity losses. Infrastructure in remote communities lacking resources for climate adaptation.

6. ECOSYSTEMS: Earth's living systems are experiencing widespread changes, and the impacts of global environmental change on the vulnerability and resilience of ecological systems are becoming increasingly clear.

Biodiversity Impacts: Species migration patterns shifting with temperature zones. Arctic species facing habitat loss as sea ice declines. Forest composition changing due to fire, drought, and pest pressures.

Ecosystem Service Valuations: Carbon sequestration capacity declining as forests experience stress. Water filtration services from wetlands threatened by sea level rise. Pollination services disrupted by changing growing seasons.

Marine Ecosystems: Ocean acidification affecting shell-forming species. Coral reef bleaching increasing in frequency and severity. Fish stock migrations affecting commercial and subsistence fishing.

REPORT 2: INTERGOVERNMENTAL PANEL ON CLIMATE CHANGE SIXTH ASSESSMENT REPORT
Working Group II: Impacts, Adaptation, and Vulnerability
Summary for Policymakers

SUMMARY FOR POLICYMAKERS

Human-induced climate change is causing dangerous and widespread disruption in nature and affecting the lives of billions of people around the world, despite efforts to reduce the risks. People and ecosystems least able to cope are being hardest hit, said scientists in the latest Intergovernmental Panel on Climate Change (IPCC) report.

OBSERVED IMPACTS AND CURRENT RISKS:

Global Temperature Impacts: Human-induced climate change has caused widespread adverse impacts and related losses and damages to nature and people, beyond natural climate variability. Global surface temperature has increased faster since 1970 than in any other 50-year period over at least the last 2000 years.

Regional Climate Changes: Climate change has caused substantial damages, and increasingly irreversible losses, in terrestrial, freshwater, and marine ecosystems. Hundreds of local losses of species have been driven by increases in the magnitude of heat extremes with some irreversible impacts.

Human Systems Impacts: Climate change has reduced food security and affected water security due to warming, changing precipitation patterns, reduction and loss of cryospheric elements, and greater frequency and intensity of extreme events.

FUTURE RISKS AND LONG-TERM IMPACTS:

Temperature Scenarios: Global warming of 1.5°C and 2°C will be exceeded during the 21st century unless deep reductions in CO2 and other greenhouse gas emissions occur in the coming decades. Continued warming will further intensify climate change impacts.

Regional Projections: 
Arctic: Up to 4°C warming by 2100 causing dramatic ice loss and ecosystem transformation. Permafrost thaw releasing stored carbon accelerating global warming.

Mediterranean: 20% reduction in precipitation coupled with 2-3°C warming creating severe drought conditions. Agricultural productivity declining 30% by 2080.

Small Islands: Complete submersion risk for 1000+ communities by 2100. Freshwater contamination and ecosystem collapse threatening habitability.

Mountain Regions: Glacier retreat accelerating with 50-80% mass loss by 2100. Alpine ecosystems facing upslope migration and habitat compression.

SECTORAL RISKS AND IMPACTS:

Water Resources: At 1.5°C warming, about 10% of global population will be exposed to severe drought. At 2°C, this increases to 17% of population. Urban water security threatened in regions dependent on snowmelt and monsoons.

Food Security: Global crop yields projecting 2-6% decline per decade due to climate change. Marine fisheries productivity declining 3-12% by 2050. Livestock heat stress reducing productivity 7-10% globally.

Human Health: Additional 250,000 deaths annually between 2030-2050 from malnutrition, malaria, diarrhea, and heat stress. Healthcare systems overwhelmed by climate-related health impacts.

Economic Systems: Global economic losses reaching 10-23% of GDP by 2100 under high warming scenarios. Developing countries facing disproportionate impacts despite contributing least to emissions.

ADAPTATION LIMITS AND RISKS:

Soft Limits: Financial, governance, institutional, and policy constraints limiting adaptation implementation. Knowledge gaps hindering effective response planning.

Hard Limits: Biophysical thresholds beyond which adaptation cannot prevent intolerable risks. Ecosystem tipping points leading to irreversible changes.

MALADAPTATION: Some adaptation responses may inadvertently increase greenhouse gas emissions or worsen equity outcomes. Careful planning required to avoid unintended consequences.

REPORT 3: DEPARTMENT OF ENERGY PATHWAYS TO COMMERCIAL LIFTOFF REPORTS
Renewable Energy Deployment Strategies and Economic Analysis

EXECUTIVE SUMMARY: RENEWABLE ENERGY TRANSFORMATION

The United States stands at a critical juncture in its energy transition, with renewable energy technologies reaching cost parity or advantage over fossil fuels in most markets. This comprehensive analysis examines pathways for accelerating renewable energy deployment to achieve net-zero emissions by 2050 while maintaining energy security and economic competitiveness.

CURRENT RENEWABLE ENERGY LANDSCAPE:

Market Penetration: Renewable energy comprised 21% of total electricity generation in 2023, led by wind (9.2%) and solar (3.4%). Hydroelectric contributes 6.1% with remaining from biomass and geothermal sources.

Cost Competitiveness: Unsubsidized utility-scale solar costs declined 90% from 2010-2023, reaching $28-41/MWh. Wind costs declined 70% over same period to $26-50/MWh. Both significantly below natural gas peaking plant costs of $175-250/MWh.

Investment Trends: Private renewable energy investment reached $384 billion in 2023, exceeding fossil fuel investment for third consecutive year. Corporate renewable energy procurement contracts totaled 23.7 GW, demonstrating sustained business demand.

DEPLOYMENT SCENARIOS AND PATHWAYS:

Baseline Scenario (Current Policies): Renewable capacity reaches 160 GW by 2030, achieving 42% of electricity generation. Emissions reduction of 28% from 2005 levels by 2030.

Accelerated Scenario (Enhanced Policies): Renewable capacity reaches 240 GW by 2030, achieving 65% of electricity generation. Emissions reduction of 52% from 2005 levels by 2030.

Transformational Scenario (Maximum Deployment): Renewable capacity reaches 320 GW by 2030, achieving 78% of electricity generation. Emissions reduction of 68% from 2005 levels by 2030.

TECHNOLOGY-SPECIFIC ANALYSIS:

Solar Energy:
Current Capacity: 131 GW installed, with 75 GW utility-scale and 56 GW distributed
Growth Potential: 1,000+ GW technical potential on suitable land
Economic Benefits: $230 billion investment opportunity creating 3.3 million jobs by 2030
Challenges: Grid integration and energy storage requirements for reliability

Wind Energy:
Current Capacity: 140 GW onshore, 42 MW offshore
Growth Potential: 2,000+ GW onshore technical potential, 4,200 GW offshore potential
Economic Benefits: $180 billion investment opportunity creating 2.8 million jobs by 2030
Challenges: Transmission infrastructure and public acceptance

Energy Storage:
Current Capacity: 8.8 GW deployed, primarily lithium-ion batteries
Growth Potential: 100+ GW needed by 2030 for grid reliability
Economic Benefits: $120 billion market opportunity with declining costs
Technology Evolution: Long-duration storage and alternative chemistries developing

ECONOMIC IMPACT ANALYSIS:

Job Creation: Renewable energy sector employs 3.3 million Americans, growing 8% annually. Solar installer and wind technician among fastest-growing occupations.

Regional Economic Development: Rural communities hosting wind and solar projects receiving $1.8 billion annually in lease payments and local taxes. Manufacturing renaissance in Midwest and Southeast.

Consumer Benefits: Renewable energy saving average household $500 annually on electricity bills. Industrial customers reporting 20-40% energy cost reductions through renewable procurement.

GRID MODERNIZATION REQUIREMENTS:

Transmission Infrastructure: $360 billion investment needed in transmission and distribution infrastructure to accommodate renewable growth. Regional transmission organizations planning 24,000 miles of new high-voltage lines.

Grid Flexibility: Smart grid technologies and demand response programs required to manage variable renewable output. Advanced forecasting and grid management systems reducing integration costs.

Interconnection Reform: Streamlining interconnection processes could reduce timeline from 5 years to 2 years, accelerating deployment and reducing costs.

POLICY RECOMMENDATIONS:

Federal Level:
- Extend and expand production tax credits and investment tax credits through 2035
- Invest $100 billion in transmission infrastructure and grid modernization
- Streamline permitting processes for renewable energy projects
- Support research and development for emerging technologies

State Level:
- Implement renewable portfolio standards targeting 50% renewable electricity by 2030
- Reform utility regulations to incentivize clean energy investments
- Develop state clean energy financing programs
- Strengthen building codes to support renewable energy integration

Local Level:
- Streamline zoning and permitting for distributed renewable energy
- Develop community solar programs for broader access
- Invest in workforce development for clean energy jobs
- Create green infrastructure requirements for new development

BARRIERS TO DEPLOYMENT:

Regulatory Challenges: Complex permitting processes averaging 3-5 years for large projects. Inconsistent policies across jurisdictions creating uncertainty.

Market Design: Wholesale electricity markets not fully recognizing renewable energy benefits. Price signals insufficient to drive optimal storage investments.

Social Acceptance: Public opposition to transmission lines and large renewable projects. Need for better community engagement and benefit-sharing programs.

Technical Challenges: Grid integration complexities as renewable penetration increases. Need for enhanced forecasting and grid management capabilities.

Based on synthesis of these three comprehensive reports, develop integrated climate policy recommendations that address:

1. What are the most critical climate risks identified across all three reports and how do they interact with each other? Provide specific quantitative evidence and timeline projections.

2. How do the economic impacts of climate change compare with the costs and benefits of renewable energy transition? Analyze the cost-benefit tradeoffs using data from all reports.

3. What policy interventions are most essential for achieving climate goals while maintaining economic growth? Prioritize recommendations based on urgency, feasibility, and impact potential.

4. How should adaptation and mitigation strategies be integrated for maximum effectiveness? Consider the interaction between climate impacts and energy system resilience.

5. What are the key implementation challenges and how can they be overcome? Address regulatory, economic, technical, and social barriers identified across the reports.

Your analysis must synthesize quantitative data, timeline projections, and policy recommendations from all three reports to provide a comprehensive climate action framework.""",
        "validator": "Response must provide integrated climate policy recommendations addressing all 5 specific questions with quantitative synthesis of climate risks, economic impacts, and renewable energy deployment data from the National Climate Assessment, IPCC report, and DOE analysis."
    },
    
    # ======================================================================================
    # CATEGORY 2: LARGE CODEBASE COMPREHENSION (2 scenarios)
    # ======================================================================================
    
    {
        "name": "E-commerce Platform Memory Leak Investigation",
        "prompt": """
CONTEXT: You are a senior software engineer investigating a critical memory leak in a large e-commerce platform. The application is experiencing gradual memory consumption increases leading to crashes during peak traffic. Below is the complete codebase for the core order processing system.

FILE: src/main/java/com/ecommerce/OrderService.java
```java
package com.ecommerce;

import java.util.*;
import java.util.concurrent.*;
import java.time.LocalDateTime;
import java.util.stream.Collectors;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class OrderService {
    
    @Autowired
    private OrderRepository orderRepository;
    
    @Autowired
    private UserService userService;
    
    @Autowired
    private InventoryService inventoryService;
    
    @Autowired
    private PaymentService paymentService;
    
    @Autowired
    private NotificationService notificationService;
    
    @Autowired
    private CacheManager cacheManager;
    
    private final Map<String, List<OrderEvent>> orderEventCache = new ConcurrentHashMap<>();
    private final Map<String, OrderProcessor> activeProcessors = new ConcurrentHashMap<>();
    private final ExecutorService executorService = Executors.newFixedThreadPool(50);
    private final List<OrderListener> globalOrderListeners = new ArrayList<>();
    
    // Static collections that grow over time
    private static final Map<String, OrderMetrics> ORDER_METRICS = new ConcurrentHashMap<>();
    private static final List<String> PROCESSED_ORDER_IDS = Collections.synchronizedList(new ArrayList<>());
    private static final Map<String, byte[]> ORDER_SNAPSHOTS = new ConcurrentHashMap<>();
    
    @Transactional
    public Order processOrder(OrderRequest request) {
        String orderId = generateOrderId();
        
        try {
            // Create order processor that maintains references
            OrderProcessor processor = new OrderProcessor(orderId, request);
            activeProcessors.put(orderId, processor);
            
            // Add to global tracking without cleanup
            PROCESSED_ORDER_IDS.add(orderId);
            
            // Cache order events indefinitely
            List<OrderEvent> events = new ArrayList<>();
            orderEventCache.put(orderId, events);
            
            // Store order snapshot for debugging (never cleaned)
            byte[] orderSnapshot = serializeOrderData(request);
            ORDER_SNAPSHOTS.put(orderId, orderSnapshot);
            
            // Validate order
            ValidationResult validation = validateOrder(request);
            if (!validation.isValid()) {
                // Processor remains in activeProcessors map even for invalid orders
                events.add(new OrderEvent("VALIDATION_FAILED", validation.getErrors()));
                throw new OrderValidationException(validation.getErrors());
            }
            events.add(new OrderEvent("VALIDATION_PASSED", null));
            
            // Check inventory
            InventoryCheckResult inventoryResult = inventoryService.checkAvailability(request.getItems());
            events.add(new OrderEvent("INVENTORY_CHECKED", inventoryResult));
            
            if (!inventoryResult.isAvailable()) {
                // Processor still not cleaned up
                events.add(new OrderEvent("INVENTORY_INSUFFICIENT", inventoryResult.getUnavailableItems()));
                throw new InsufficientInventoryException(inventoryResult.getUnavailableItems());
            }
            
            // Process payment
            PaymentResult paymentResult = paymentService.processPayment(request.getPaymentInfo(), request.getTotalAmount());
            events.add(new OrderEvent("PAYMENT_PROCESSED", paymentResult));
            
            if (!paymentResult.isSuccessful()) {
                events.add(new OrderEvent("PAYMENT_FAILED", paymentResult.getErrorMessage()));
                throw new PaymentProcessingException(paymentResult.getErrorMessage());
            }
            
            // Create order entity
            Order order = new Order();
            order.setId(orderId);
            order.setUserId(request.getUserId());
            order.setItems(request.getItems());
            order.setTotalAmount(request.getTotalAmount());
            order.setStatus(OrderStatus.PROCESSING);
            order.setCreatedAt(LocalDateTime.now());
            
            // Save order
            Order savedOrder = orderRepository.save(order);
            events.add(new OrderEvent("ORDER_SAVED", savedOrder));
            
            // Update inventory
            inventoryService.reserveItems(request.getItems());
            events.add(new OrderEvent("INVENTORY_RESERVED", null));
            
            // Schedule fulfillment asynchronously
            executorService.submit(() -> {
                try {
                    fulfillOrder(savedOrder);
                } catch (Exception e) {
                    handleFulfillmentError(savedOrder, e);
                }
            });
            
            // Update metrics (grows indefinitely)
            OrderMetrics metrics = ORDER_METRICS.computeIfAbsent(orderId, k -> new OrderMetrics());
            metrics.incrementProcessedCount();
            metrics.setLastProcessedAt(LocalDateTime.now());
            
            // Notify listeners
            notifyOrderListeners(savedOrder, events);
            
            // Cache management issue - never cleanup old entries
            cacheManager.cacheOrder(savedOrder);
            
            return savedOrder;
            
        } catch (Exception e) {
            // Even in exception cases, cleanup is not performed
            List<OrderEvent> events = orderEventCache.get(orderId);
            if (events != null) {
                events.add(new OrderEvent("ERROR_OCCURRED", e.getMessage()));
            }
            
            // Log error but don't cleanup resources
            logOrderError(orderId, e);
            throw e;
        }
        // Note: activeProcessors.remove(orderId) is never called
    }
    
    private String generateOrderId() {
        return "ORD-" + System.currentTimeMillis() + "-" + UUID.randomUUID().toString().substring(0, 8);
    }
    
    private byte[] serializeOrderData(OrderRequest request) {
        // This creates large byte arrays that accumulate in ORDER_SNAPSHOTS
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(request);
            oos.close();
            return baos.toByteArray();
        } catch (Exception e) {
            return new byte[0];
        }
    }
    
    private ValidationResult validateOrder(OrderRequest request) {
        ValidationResult result = new ValidationResult();
        
        if (request.getUserId() == null || request.getUserId().trim().isEmpty()) {
            result.addError("User ID is required");
        }
        
        if (request.getItems() == null || request.getItems().isEmpty()) {
            result.addError("Order must contain at least one item");
        }
        
        if (request.getTotalAmount() == null || request.getTotalAmount().compareTo(BigDecimal.ZERO) <= 0) {
            result.addError("Total amount must be greater than zero");
        }
        
        // Validate each item
        if (request.getItems() != null) {
            for (OrderItem item : request.getItems()) {
                if (item.getProductId() == null) {
                    result.addError("Product ID is required for all items");
                }
                if (item.getQuantity() <= 0) {
                    result.addError("Quantity must be greater than zero for product: " + item.getProductId());
                }
                if (item.getPrice() == null || item.getPrice().compareTo(BigDecimal.ZERO) <= 0) {
                    result.addError("Price must be greater than zero for product: " + item.getProductId());
                }
            }
        }
        
        return result;
    }
    
    private void fulfillOrder(Order order) {
        try {
            // Create fulfillment processor that's never cleaned up
            FulfillmentProcessor fulfillmentProcessor = new FulfillmentProcessor(order);
            String key = "fulfillment-" + order.getId();
            activeProcessors.put(key, fulfillmentProcessor);
            
            List<OrderEvent> events = orderEventCache.get(order.getId());
            if (events != null) {
                events.add(new OrderEvent("FULFILLMENT_STARTED", null));
            }
            
            // Simulate fulfillment process
            Thread.sleep(2000);
            
            // Update order status
            order.setStatus(OrderStatus.FULFILLED);
            order.setFulfilledAt(LocalDateTime.now());
            orderRepository.save(order);
            
            if (events != null) {
                events.add(new OrderEvent("FULFILLMENT_COMPLETED", null));
            }
            
            // Send notification
            notificationService.sendOrderFulfilledNotification(order);
            
            // Update metrics again
            OrderMetrics metrics = ORDER_METRICS.get(order.getId());
            if (metrics != null) {
                metrics.incrementFulfilledCount();
                metrics.setLastFulfilledAt(LocalDateTime.now());
            }
            
            // Note: fulfillmentProcessor is never removed from activeProcessors
            
        } catch (Exception e) {
            handleFulfillmentError(order, e);
        }
    }
    
    private void handleFulfillmentError(Order order, Exception e) {
        List<OrderEvent> events = orderEventCache.get(order.getId());
        if (events != null) {
            events.add(new OrderEvent("FULFILLMENT_ERROR", e.getMessage()));
        }
        
        order.setStatus(OrderStatus.FAILED);
        orderRepository.save(order);
        
        notificationService.sendOrderFailedNotification(order, e.getMessage());
        logOrderError(order.getId(), e);
    }
    
    private void logOrderError(String orderId, Exception e) {
        // Create error log entry that accumulates
        ErrorLog errorLog = new ErrorLog(orderId, e.getMessage(), LocalDateTime.now());
        // This grows indefinitely as well
        globalOrderListeners.add(new ErrorLogListener(errorLog));
    }
    
    private void notifyOrderListeners(Order order, List<OrderEvent> events) {
        for (OrderListener listener : globalOrderListeners) {
            try {
                listener.onOrderProcessed(order, events);
            } catch (Exception e) {
                // Ignore listener errors but don't remove failed listeners
            }
        }
    }
    
    public void addOrderListener(OrderListener listener) {
        globalOrderListeners.add(listener);
        // No mechanism to remove listeners
    }
    
    public OrderStatus getOrderStatus(String orderId) {
        Order order = orderRepository.findById(orderId);
        return order != null ? order.getStatus() : null;
    }
    
    public List<OrderEvent> getOrderEvents(String orderId) {
        return orderEventCache.get(orderId);
    }
    
    public OrderMetrics getOrderMetrics(String orderId) {
        return ORDER_METRICS.get(orderId);
    }
    
    // Administrative methods that don't clean up properly
    public void cancelOrder(String orderId) {
        Order order = orderRepository.findById(orderId);
        if (order != null && order.getStatus() == OrderStatus.PROCESSING) {
            order.setStatus(OrderStatus.CANCELLED);
            orderRepository.save(order);
            
            List<OrderEvent> events = orderEventCache.get(orderId);
            if (events != null) {
                events.add(new OrderEvent("ORDER_CANCELLED", null));
            }
            
            // Release inventory
            inventoryService.releaseReservedItems(order.getItems());
            
            // Refund payment
            paymentService.refundPayment(order.getTotalAmount());
            
            // Send notification
            notificationService.sendOrderCancelledNotification(order);
        }
        // Note: No cleanup of caches or processors
    }
    
    // Problematic method that creates temporary data structures
    public List<Order> getOrderHistory(String userId, int limit) {
        List<Order> allOrders = orderRepository.findByUserId(userId);
        
        // Create temporary maps that may not be garbage collected efficiently
        Map<String, Order> orderMap = new HashMap<>();
        Map<String, List<OrderEvent>> eventMap = new HashMap<>();
        
        for (Order order : allOrders) {
            orderMap.put(order.getId(), order);
            List<OrderEvent> events = orderEventCache.get(order.getId());
            if (events != null) {
                eventMap.put(order.getId(), new ArrayList<>(events)); // Defensive copy
            }
        }
        
        // Sort and limit results
        return allOrders.stream()
                .sorted((o1, o2) -> o2.getCreatedAt().compareTo(o1.getCreatedAt()))
                .limit(limit)
                .collect(Collectors.toList());
    }
    
    // Method that creates large temporary objects
    public OrderReport generateOrderReport(String userId, LocalDateTime startDate, LocalDateTime endDate) {
        List<Order> orders = orderRepository.findByUserIdAndDateRange(userId, startDate, endDate);
        
        // Create large temporary data structures
        Map<String, BigDecimal> dailyTotals = new HashMap<>();
        Map<String, Integer> productCounts = new HashMap<>();
        Map<String, List<OrderEvent>> eventsByDate = new HashMap<>();
        
        for (Order order : orders) {
            String dateKey = order.getCreatedAt().toLocalDate().toString();
            
            dailyTotals.merge(dateKey, order.getTotalAmount(), BigDecimal::add);
            
            for (OrderItem item : order.getItems()) {
                productCounts.merge(item.getProductId(), item.getQuantity(), Integer::sum);
            }
            
            List<OrderEvent> events = orderEventCache.get(order.getId());
            if (events != null) {
                eventsByDate.computeIfAbsent(dateKey, k -> new ArrayList<>()).addAll(events);
            }
        }
        
        OrderReport report = new OrderReport();
        report.setUserId(userId);
        report.setStartDate(startDate);
        report.setEndDate(endDate);
        report.setDailyTotals(dailyTotals);
        report.setProductCounts(productCounts);
        report.setTotalOrders(orders.size());
        report.setTotalAmount(orders.stream()
                .map(Order::getTotalAmount)
                .reduce(BigDecimal.ZERO, BigDecimal::add));
        
        return report;
    }
}
```

FILE: src/main/java/com/ecommerce/OrderProcessor.java
```java
package com.ecommerce;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.*;

public class OrderProcessor {
    private final String orderId;
    private final OrderRequest originalRequest;
    private final Map<String, Object> processingContext;
    private final List<ProcessingStep> executedSteps;
    private final ExecutorService stepExecutor;
    private final Map<String, Future<?>> activeTasks;
    
    // Large data structures that accumulate over time
    private final List<byte[]> processingSnapshots;
    private final Map<String, String> debugInformation;
    private final List<Exception> encounteredErrors;
    
    public OrderProcessor(String orderId, OrderRequest request) {
        this.orderId = orderId;
        this.originalRequest = request;
        this.processingContext = new ConcurrentHashMap<>();
        this.executedSteps = Collections.synchronizedList(new ArrayList<>());
        this.stepExecutor = Executors.newFixedThreadPool(10);
        this.activeTasks = new ConcurrentHashMap<>();
        this.processingSnapshots = Collections.synchronizedList(new ArrayList<>());
        this.debugInformation = new ConcurrentHashMap<>();
        this.encounteredErrors = Collections.synchronizedList(new ArrayList<>());
        
        // Initialize processing context with large data
        initializeProcessingContext();
    }
    
    private void initializeProcessingContext() {
        processingContext.put("startTime", LocalDateTime.now());
        processingContext.put("orderId", orderId);
        processingContext.put("originalRequest", originalRequest);
        
        // Store large debug information that never gets cleaned
        debugInformation.put("creationTimestamp", String.valueOf(System.nanoTime()));
        debugInformation.put("threadInfo", Thread.currentThread().toString());
        debugInformation.put("memoryUsage", String.valueOf(Runtime.getRuntime().totalMemory()));
        
        // Create initial processing snapshot
        takeProcessingSnapshot("INITIALIZATION");
    }
    
    public void executeStep(ProcessingStep step) {
        try {
            executedSteps.add(step);
            processingContext.put("currentStep", step.getName());
            
            // Take snapshot before execution
            takeProcessingSnapshot("BEFORE_" + step.getName());
            
            Future<?> task = stepExecutor.submit(() -> {
                try {
                    step.execute(processingContext);
                    debugInformation.put(step.getName() + "_completed", LocalDateTime.now().toString());
                } catch (Exception e) {
                    encounteredErrors.add(e);
                    debugInformation.put(step.getName() + "_error", e.getMessage());
                    throw new RuntimeException(e);
                }
            });
            
            activeTasks.put(step.getName(), task);
            
            // Take snapshot after execution
            takeProcessingSnapshot("AFTER_" + step.getName());
            
        } catch (Exception e) {
            encounteredErrors.add(e);
            debugInformation.put("step_execution_error", e.getMessage());
        }
    }
    
    private void takeProcessingSnapshot(String phase) {
        try {
            // Create large snapshot data that accumulates
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            
            Map<String, Object> snapshot = new HashMap<>();
            snapshot.put("phase", phase);
            snapshot.put("timestamp", LocalDateTime.now());
            snapshot.put("context", new HashMap<>(processingContext));
            snapshot.put("executedSteps", new ArrayList<>(executedSteps));
            snapshot.put("debugInfo", new HashMap<>(debugInformation));
            
            oos.writeObject(snapshot);
            oos.close();
            
            processingSnapshots.add(baos.toByteArray());
            
        } catch (Exception e) {
            encounteredErrors.add(e);
        }
    }
    
    public boolean isComplete() {
        return activeTasks.values().stream().allMatch(Future::isDone);
    }
    
    public List<Exception> getErrors() {
        return new ArrayList<>(encounteredErrors);
    }
    
    public Map<String, Object> getProcessingContext() {
        return new HashMap<>(processingContext);
    }
    
    // Method that should clean up resources but doesn't
    public void cleanup() {
        // This method exists but doesn't actually clean up properly
        debugInformation.put("cleanup_called", LocalDateTime.now().toString());
        // stepExecutor.shutdown() is never called
        // activeTasks is never cleared
        // processingSnapshots is never cleared
        // encounteredErrors is never cleared
    }
}
```

FILE: src/main/java/com/ecommerce/CacheManager.java
```java
package com.ecommerce;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class CacheManager {
    
    // Multiple cache layers that grow indefinitely
    private final Map<String, Order> orderCache = new ConcurrentHashMap<>();
    private final Map<String, User> userCache = new ConcurrentHashMap<>();
    private final Map<String, List<OrderItem>> orderItemsCache = new ConcurrentHashMap<>();
    private final Map<String, PaymentInfo> paymentCache = new ConcurrentHashMap<>();
    private final Map<String, LocalDateTime> cacheTimestamps = new ConcurrentHashMap<>();
    
    // Secondary caches that duplicate data
    private final Map<String, String> orderStatusCache = new ConcurrentHashMap<>();
    private final Map<String, BigDecimal> orderTotalCache = new ConcurrentHashMap<>();
    private final Map<String, String> userEmailCache = new ConcurrentHashMap<>();
    
    // Large lookup tables that never expire
    private final Map<String, Set<String>> userOrdersLookup = new ConcurrentHashMap<>();
    private final Map<String, Set<String>> productOrdersLookup = new ConcurrentHashMap<>();
    
    public void cacheOrder(Order order) {
        String orderId = order.getId();
        
        // Cache the full order
        orderCache.put(orderId, order);
        cacheTimestamps.put(orderId, LocalDateTime.now());
        
        // Cache individual components (data duplication)
        orderStatusCache.put(orderId, order.getStatus().toString());
        orderTotalCache.put(orderId, order.getTotalAmount());
        orderItemsCache.put(orderId, new ArrayList<>(order.getItems()));
        
        // Update lookup tables (these grow indefinitely)
        String userId = order.getUserId();
        userOrdersLookup.computeIfAbsent(userId, k -> new HashSet<>()).add(orderId);
        
        for (OrderItem item : order.getItems()) {
            productOrdersLookup.computeIfAbsent(item.getProductId(), k -> new HashSet<>()).add(orderId);
        }
        
        // Cache user information if available
        if (order.getUser() != null) {
            userCache.put(userId, order.getUser());
            userEmailCache.put(userId, order.getUser().getEmail());
        }
        
        // Cache payment information (sensitive data retention issue)
        if (order.getPaymentInfo() != null) {
            paymentCache.put(orderId, order.getPaymentInfo());
        }
    }
    
    public Order getCachedOrder(String orderId) {
        return orderCache.get(orderId);
    }
    
    public List<String> getCachedOrdersForUser(String userId) {
        Set<String> orderIds = userOrdersLookup.get(userId);
        return orderIds != null ? new ArrayList<>(orderIds) : new ArrayList<>();
    }
    
    public List<String> getCachedOrdersForProduct(String productId) {
        Set<String> orderIds = productOrdersLookup.get(productId);
        return orderIds != null ? new ArrayList<>(orderIds) : new ArrayList<>();
    }
    
    // Method that should implement cache eviction but doesn't
    public void evictOldEntries() {
        // This method exists but doesn't actually remove anything
        LocalDateTime cutoff = LocalDateTime.now().minusHours(24);
        
        // Count old entries but don't remove them
        long oldEntries = cacheTimestamps.entrySet().stream()
                .mapToLong(entry -> entry.getValue().isBefore(cutoff) ? 1 : 0)
                .sum();
        
        System.out.println("Found " + oldEntries + " old cache entries, but not removing them");
        // Actual eviction logic is commented out or never implemented
    }
    
    public void clearCache() {
        // Administrative method that should clear cache but has issues
        System.out.println("Attempting to clear cache...");
        
        // Only clears some caches, not all
        orderCache.clear();
        userCache.clear();
        // orderItemsCache, paymentCache, lookup tables, etc. are never cleared
        
        System.out.println("Cache cleared (partially)");
    }
    
    public Map<String, Integer> getCacheStatistics() {
        Map<String, Integer> stats = new HashMap<>();
        stats.put("orderCache", orderCache.size());
        stats.put("userCache", userCache.size());
        stats.put("orderItemsCache", orderItemsCache.size());
        stats.put("paymentCache", paymentCache.size());
        stats.put("orderStatusCache", orderStatusCache.size());
        stats.put("orderTotalCache", orderTotalCache.size());
        stats.put("userEmailCache", userEmailCache.size());
        stats.put("userOrdersLookup", userOrdersLookup.size());
        stats.put("productOrdersLookup", productOrdersLookup.size());
        stats.put("cacheTimestamps", cacheTimestamps.size());
        return stats;
    }
}
```

FILE: src/main/java/com/ecommerce/FulfillmentProcessor.java
```java
package com.ecommerce;

import java.util.*;
import java.util.concurrent.*;
import java.time.LocalDateTime;

public class FulfillmentProcessor extends OrderProcessor {
    
    private final Order order;
    private final Map<String, WarehouseTask> warehouseTasks;
    private final List<ShippingLabel> generatedLabels;
    private final Map<String, TrackingInfo> trackingInformation;
    private final ExecutorService warehouseExecutor;
    
    // Large collections that accumulate fulfillment data
    private final List<InventoryTransaction> inventoryTransactions;
    private final Map<String, PackingSlip> packingSlips;
    private final List<QualityCheckResult> qualityChecks;
    private final Map<String, byte[]> documentStorage;
    
    public FulfillmentProcessor(Order order) {
        super("fulfillment-" + order.getId(), null);
        this.order = order;
        this.warehouseTasks = new ConcurrentHashMap<>();
        this.generatedLabels = Collections.synchronizedList(new ArrayList<>());
        this.trackingInformation = new ConcurrentHashMap<>();
        this.warehouseExecutor = Executors.newFixedThreadPool(20);
        this.inventoryTransactions = Collections.synchronizedList(new ArrayList<>());
        this.packingSlips = new ConcurrentHashMap<>();
        this.qualityChecks = Collections.synchronizedList(new ArrayList<>());
        this.documentStorage = new ConcurrentHashMap<>();
        
        initializeFulfillmentProcess();
    }
    
    private void initializeFulfillmentProcess() {
        // Create warehouse tasks for each item
        for (OrderItem item : order.getItems()) {
            WarehouseTask task = new WarehouseTask(
                order.getId(), 
                item.getProductId(), 
                item.getQuantity()
            );
            warehouseTasks.put(item.getProductId(), task);
            
            // Submit task to executor but never track completion properly
            warehouseExecutor.submit(() -> processWarehouseTask(task));
        }
        
        // Generate initial documents that accumulate in memory
        generatePackingSlips();
        generateShippingLabels();
    }
    
    private void processWarehouseTask(WarehouseTask task) {
        try {
            // Simulate warehouse operations
            Thread.sleep(1000);
            
            // Record inventory transaction (these accumulate)
            InventoryTransaction transaction = new InventoryTransaction(
                task.getProductId(),
                task.getQuantity(),
                "FULFILLED",
                LocalDateTime.now()
            );
            inventoryTransactions.add(transaction);
            
            // Perform quality check and store results
            QualityCheckResult qualityCheck = performQualityCheck(task);
            qualityChecks.add(qualityCheck);
            
            // Generate tracking information
            String trackingNumber = generateTrackingNumber();
            TrackingInfo tracking = new TrackingInfo(trackingNumber, order.getId(), task.getProductId());
            trackingInformation.put(task.getProductId(), tracking);
            
            task.setStatus(TaskStatus.COMPLETED);
            task.setCompletedAt(LocalDateTime.now());
            
        } catch (Exception e) {
            task.setStatus(TaskStatus.FAILED);
            task.setErrorMessage(e.getMessage());
        }
    }
    
    private void generatePackingSlips() {
        for (OrderItem item : order.getItems()) {
            PackingSlip slip = new PackingSlip();
            slip.setOrderId(order.getId());
            slip.setProductId(item.getProductId());
            slip.setQuantity(item.getQuantity());
            slip.setGeneratedAt(LocalDateTime.now());
            
            packingSlips.put(item.getProductId(), slip);
            
            // Store as document (large binary data that accumulates)
            byte[] slipDocument = generatePackingSlipDocument(slip);
            documentStorage.put("packing_slip_" + item.getProductId(), slipDocument);
        }
    }
    
    private void generateShippingLabels() {
        for (OrderItem item : order.getItems()) {
            ShippingLabel label = new ShippingLabel();
            label.setOrderId(order.getId());
            label.setProductId(item.getProductId());
            label.setDestinationAddress(order.getShippingAddress());
            label.setGeneratedAt(LocalDateTime.now());
            
            generatedLabels.add(label);
            
            // Store label as binary document
            byte[] labelDocument = generateLabelDocument(label);
            documentStorage.put("shipping_label_" + item.getProductId(), labelDocument);
        }
    }
    
    private QualityCheckResult performQualityCheck(WarehouseTask task) {
        // Simulate quality check process
        QualityCheckResult result = new QualityCheckResult();
        result.setProductId(task.getProductId());
        result.setOrderId(task.getOrderId());
        result.setCheckTime(LocalDateTime.now());
        result.setPassed(Math.random() > 0.05); // 95% pass rate
        
        if (!result.isPassed()) {
            result.setFailureReason("Quality standard not met");
        }
        
        return result;
    }
    
    private String generateTrackingNumber() {
        return "TRK-" + System.currentTimeMillis() + "-" + UUID.randomUUID().toString().substring(0, 8);
    }
    
    private byte[] generatePackingSlipDocument(PackingSlip slip) {
        // Generate large document that gets stored in memory
        StringBuilder doc = new StringBuilder();
        doc.append("PACKING SLIP\\n");
        doc.append("Order ID: ").append(slip.getOrderId()).append("\\n");
        doc.append("Product ID: ").append(slip.getProductId()).append("\\n");
        doc.append("Quantity: ").append(slip.getQuantity()).append("\\n");
        doc.append("Generated: ").append(slip.getGeneratedAt()).append("\\n");
        
        // Add lots of padding to make document larger
        for (int i = 0; i < 1000; i++) {
            doc.append("Packing instruction line ").append(i).append("\\n");
        }
        
        return doc.toString().getBytes();
    }
    
    private byte[] generateLabelDocument(ShippingLabel label) {
        // Generate large label document
        StringBuilder doc = new StringBuilder();
        doc.append("SHIPPING LABEL\\n");
        doc.append("Order ID: ").append(label.getOrderId()).append("\\n");
        doc.append("Product ID: ").append(label.getProductId()).append("\\n");
        doc.append("Destination: ").append(label.getDestinationAddress()).append("\\n");
        doc.append("Generated: ").append(label.getGeneratedAt()).append("\\n");
        
        // Add barcode and other large data
        for (int i = 0; i < 500; i++) {
            doc.append("Barcode data segment ").append(i).append("\\n");
        }
        
        return doc.toString().getBytes();
    }
    
    public boolean isAllTasksComplete() {
        return warehouseTasks.values().stream()
                .allMatch(task -> task.getStatus() == TaskStatus.COMPLETED || task.getStatus() == TaskStatus.FAILED);
    }
    
    public Map<String, Object> getFulfillmentSummary() {
        Map<String, Object> summary = new HashMap<>();
        summary.put("orderId", order.getId());
        summary.put("totalTasks", warehouseTasks.size());
        summary.put("completedTasks", warehouseTasks.values().stream().mapToLong(task -> 
            task.getStatus() == TaskStatus.COMPLETED ? 1 : 0).sum());
        summary.put("failedTasks", warehouseTasks.values().stream().mapToLong(task -> 
            task.getStatus() == TaskStatus.FAILED ? 1 : 0).sum());
        summary.put("generatedLabels", generatedLabels.size());
        summary.put("packingSlips", packingSlips.size());
        summary.put("qualityChecks", qualityChecks.size());
        summary.put("inventoryTransactions", inventoryTransactions.size());
        summary.put("documentsStored", documentStorage.size());
        
        return summary;
    }
    
    // This should clean up resources but doesn't
    @Override
    public void cleanup() {
        super.cleanup();
        // warehouseExecutor.shutdown() is never called
        // Large collections are never cleared
        // documentStorage with binary data is never cleaned
    }
}
```

Based on your analysis of this complete e-commerce order processing codebase, identify the root causes of the memory leak and provide specific recommendations for fixing them. Your analysis must include:

1. What are the primary sources of memory leaks in this codebase? Identify specific data structures, collections, and objects that are accumulating without proper cleanup.

2. Which classes and methods are the main contributors to memory consumption? Rank them by severity and explain the mechanisms causing memory retention.

3. What design patterns and coding practices are causing these memory management issues? Identify anti-patterns and architectural problems.

4. How would you fix each identified memory leak? Provide specific code changes and cleanup strategies that would resolve the issues while maintaining functionality.

5. What monitoring and prevention strategies would you implement to detect and prevent similar memory leaks in the future?

Your analysis must demonstrate deep understanding of Java memory management, garbage collection, and the specific ways these code patterns prevent proper memory cleanup.""",
        "validator": "Response must identify the specific memory leak sources in OrderService.java, OrderProcessor.java, CacheManager.java, and FulfillmentProcessor.java, explain why they cause leaks, and provide concrete fixes for each identified issue."
    },
    
    {
        "name": "Distributed System Data Flow Analysis",
        "prompt": """
CONTEXT: You are analyzing a complex microservices architecture experiencing data inconsistency issues. Below is the complete codebase for the core data processing pipeline.

FILE: user-service/src/main/java/UserController.java
```java
@RestController
public class UserController {
    @Autowired UserService userService;
    @Autowired MessageQueue messageQueue;
    
    @PostMapping("/users")
    public ResponseEntity<User> createUser(@RequestBody CreateUserRequest request) {
        User user = userService.createUser(request);
        messageQueue.publishUserCreated(user); // Async event
        return ResponseEntity.ok(user);
    }
    
    @PutMapping("/users/{id}")
    public ResponseEntity<User> updateUser(@PathVariable String id, @RequestBody UpdateUserRequest request) {
        User user = userService.updateUser(id, request);
        messageQueue.publishUserUpdated(user); // Event published before DB commit
        return ResponseEntity.ok(user);
    }
}
```

FILE: order-service/src/main/java/OrderEventHandler.java
```java
@Component
public class OrderEventHandler {
    @Autowired OrderService orderService;
    @Autowired UserServiceClient userServiceClient;
    
    @EventListener
    public void handleUserCreated(UserCreatedEvent event) {
        // Cache user data locally but may be stale
        orderService.cacheUserData(event.getUser());
    }
    
    @EventListener  
    public void handleUserUpdated(UserUpdatedEvent event) {
        // Update local cache but timing issues cause inconsistency
        orderService.updateCachedUser(event.getUser());
        
        // Process pending orders for this user
        List<Order> pendingOrders = orderService.getPendingOrdersForUser(event.getUserId());
        for (Order order : pendingOrders) {
            // Uses potentially stale user data
            orderService.validateOrderWithUserData(order, event.getUser());
        }
    }
}
```

FILE: inventory-service/src/main/java/InventoryService.java  
```java
@Service
public class InventoryService {
    @Autowired ProductRepository productRepository;
    
    @Transactional
    public boolean reserveItems(List<OrderItem> items) {
        for (OrderItem item : items) {
            Product product = productRepository.findById(item.getProductId());
            if (product.getStockLevel() >= item.getQuantity()) {
                product.setStockLevel(product.getStockLevel() - item.getQuantity());
                productRepository.save(product);
                // Race condition: Multiple concurrent reservations can oversell
            } else {
                throw new InsufficientStockException();
            }
        }
        return true;
    }
}
```

Based on this microservices codebase, identify the data consistency issues and explain how the distributed system's data flow creates these problems. Analyze the timing dependencies, race conditions, and eventual consistency challenges.""",
        "validator": "Response must identify specific data consistency issues in the user-service, order-service, and inventory-service interactions, explain the race conditions and timing problems, and describe how the distributed data flow creates inconsistencies."
    },
    
    # ======================================================================================
    # CATEGORY 3: NEEDLE IN A HAYSTACK (2 scenarios)
    # ======================================================================================
    
    {
        "name": "Security Log Analysis for Breach Detection",
        "prompt": """
CONTEXT: You are a cybersecurity analyst investigating a potential data breach. Below are extensive security logs from multiple systems over a 72-hour period. Hidden within these logs is evidence of a sophisticated attack.

SYSTEM LOGS - DAY 1 (January 15, 2024)
00:01:23 [INFO] AUTH_SVC: User login successful - user_id: emp_4521, ip: 192.168.1.45, session: sess_891023
00:01:45 [INFO] WEB_SVC: HTTP GET /dashboard - user: emp_4521, status: 200, latency: 245ms
00:02:12 [INFO] DB_SVC: Query executed - table: users, operation: SELECT, rows: 1, duration: 15ms
00:02:34 [WARN] FIREWALL: Blocked connection attempt - src: 185.220.101.23, dst: 10.0.0.50, port: 22
00:03:01 [INFO] AUTH_SVC: User login successful - user_id: emp_7832, ip: 192.168.1.67, session: sess_891024
00:03:28 [INFO] WEB_SVC: HTTP POST /api/reports - user: emp_7832, status: 201, latency: 892ms
00:04:15 [INFO] DB_SVC: Query executed - table: reports, operation: INSERT, rows: 1, duration: 67ms
00:04:42 [INFO] FILE_SVC: File uploaded - user: emp_7832, filename: quarterly_analysis.pdf, size: 2.3MB
00:05:09 [WARN] IDS: Suspicious pattern detected - multiple failed login attempts from 203.0.113.44
00:05:36 [INFO] AUTH_SVC: User login failed - username: admin, ip: 203.0.113.44, reason: invalid_password
00:06:03 [INFO] AUTH_SVC: User login failed - username: administrator, ip: 203.0.113.44, reason: invalid_password
00:06:30 [INFO] AUTH_SVC: User login failed - username: root, ip: 203.0.113.44, reason: invalid_password
00:06:57 [WARN] FIREWALL: Rate limiting activated for IP 203.0.113.44 - too many connection attempts
00:07:24 [INFO] BACKUP_SVC: Scheduled backup started - dataset: user_profiles, size: 15.7GB
00:08:11 [INFO] EMAIL_SVC: Email sent - to: manager@company.com, subject: Daily Security Report, status: delivered
00:08:38 [INFO] AUTH_SVC: User login successful - user_id: emp_2156, ip: 192.168.1.123, session: sess_891025
00:09:05 [INFO] WEB_SVC: HTTP GET /profile - user: emp_2156, status: 200, latency: 156ms
00:09:32 [INFO] DB_SVC: Query executed - table: profiles, operation: SELECT, rows: 1, duration: 22ms
00:09:59 [INFO] VPN_SVC: VPN connection established - user: emp_9845, client_ip: 98.139.183.24, server_ip: 10.0.0.100
00:10:26 [INFO] WEB_SVC: HTTP GET /documents - user: emp_9845, status: 200, latency: 334ms
00:10:53 [INFO] DB_SVC: Query executed - table: documents, operation: SELECT, rows: 247, duration: 89ms
00:11:20 [WARN] AV_SVC: Malware scan completed - files_scanned: 1247, threats_found: 0, duration: 456s
00:11:47 [INFO] AUTH_SVC: User logout - user_id: emp_4521, session: sess_891023, duration: 10m24s
00:12:14 [INFO] WEB_SVC: HTTP POST /api/timesheet - user: emp_2156, status: 201, latency: 278ms
00:12:41 [INFO] DB_SVC: Query executed - table: timesheets, operation: INSERT, rows: 1, duration: 34ms
00:13:08 [INFO] LDAP_SVC: Directory sync completed - users_updated: 23, groups_updated: 5, errors: 0
00:13:35 [WARN] FIREWALL: Blocked connection attempt - src: 194.195.240.91, dst: 10.0.0.25, port: 3389
00:14:02 [INFO] AUTH_SVC: User login successful - user_id: emp_3397, ip: 192.168.1.89, session: sess_891026
00:14:29 [INFO] WEB_SVC: HTTP GET /calendar - user: emp_3397, status: 200, latency: 198ms
00:14:56 [INFO] DB_SVC: Query executed - table: events, operation: SELECT, rows: 34, duration: 28ms
00:15:23 [INFO] PRINTER_SVC: Print job submitted - user: emp_3397, document: meeting_agenda.docx, pages: 3
00:15:50 [INFO] FILE_SVC: File access - user: emp_3397, filename: budget_2024.xlsx, operation: read
00:16:17 [WARN] DLP: Data transfer flagged - user: emp_7832, file: quarterly_analysis.pdf, destination: external_email
00:16:44 [INFO] EMAIL_SVC: Email blocked by DLP - sender: emp_7832, recipient: competitor@rival.com, attachment: quarterly_analysis.pdf
00:17:11 [INFO] INCIDENT_SVC: Incident created - id: INC_001547, type: dlp_violation, severity: medium, assigned_to: sec_team
00:17:38 [INFO] AUTH_SVC: User login successful - user_id: temp_contractor_x7k9, ip: 10.50.0.15, session: sess_891027
00:18:05 [INFO] WEB_SVC: HTTP GET /projects - user: temp_contractor_x7k9, status: 200, latency: 423ms
00:18:32 [INFO] DB_SVC: Query executed - table: projects, operation: SELECT, rows: 156, duration: 78ms
00:18:59 [WARN] ACCESS_CONTROL: Elevated privilege request - user: temp_contractor_x7k9, resource: admin_panel, status: denied
00:19:26 [INFO] VPN_SVC: VPN connection terminated - user: emp_9845, duration: 9m0s, data_transferred: 45.2MB
00:19:53 [INFO] BACKUP_SVC: Backup progress - dataset: user_profiles, progress: 23%, eta: 47m
00:20:20 [INFO] AUTH_SVC: User login successful - user_id: emp_5678, ip: 192.168.1.201, session: sess_891028
00:20:47 [INFO] WEB_SVC: HTTP GET /dashboard - user: emp_5678, status: 200, latency: 267ms
00:21:14 [INFO] DB_SVC: Query executed - table: dashboard_widgets, operation: SELECT, rows: 12, duration: 19ms
00:21:41 [INFO] MONITOR_SVC: System health check - cpu: 45%, memory: 67%, disk: 23%, network: normal
00:22:08 [WARN] FIREWALL: Port scan detected - src: 159.203.245.12, target: 10.0.0.0/24, ports: 22,80,443,3389
00:22:35 [INFO] IDS: Automated response triggered - blocking IP 159.203.245.12 for 24 hours
00:23:02 [INFO] AUTH_SVC: User logout - user_id: emp_2156, session: sess_891025, duration: 13m57s
00:23:29 [INFO] CLEANUP_SVC: Session cleanup completed - expired_sessions: 8, cache_cleared: 234MB
00:23:56 [INFO] AUTH_SVC: User login successful - user_id: emp_1234, ip: 192.168.1.78, session: sess_891029
00:24:23 [INFO] WEB_SVC: HTTP GET /reports - user: emp_1234, status: 200, latency: 445ms
00:24:50 [INFO] DB_SVC: Query executed - table: reports, operation: SELECT, rows: 89, duration: 156ms
00:25:17 [INFO] AUDIT_SVC: Audit log rotation - old_logs_archived: 156, new_log_started: audit_20240115.log
00:25:44 [WARN] CERT_SVC: Certificate expiring soon - domain: api.internal.company.com, expires: 2024-01-22, days_remaining: 7
00:26:11 [INFO] DNS_SVC: DNS query - domain: github.com, client: 192.168.1.78, response: 140.82.112.4
00:26:38 [INFO] PROXY_SVC: HTTP request forwarded - client: 192.168.1.78, url: https://github.com/company/repo, status: 200
00:27:05 [INFO] FILE_SVC: File downloaded - user: emp_1234, filename: project_specs.zip, size: 12.4MB, source: github.com
00:27:32 [WARN] AV_SVC: File scan in progress - filename: project_specs.zip, estimated_completion: 00:28:15
00:27:59 [INFO] AUTH_SVC: User login successful - user_id: sys_backup, ip: 127.0.0.1, session: sess_891030
00:28:26 [INFO] BACKUP_SVC: System backup initiated - user: sys_backup, target: /data/critical, size: 234GB
00:28:53 [INFO] AV_SVC: File scan completed - filename: project_specs.zip, result: clean, duration: 54s
00:29:20 [INFO] EMAIL_SVC: Email sent - to: team-leads@company.com, subject: Weekly Project Update, status: delivered
00:29:47 [WARN] DISK_SVC: Disk space warning - partition: /var/log, usage: 78%, threshold: 75%
00:30:14 [INFO] MAINTENANCE_SVC: Log rotation scheduled - target: /var/log/application.log, action: compress_and_archive
00:30:41 [INFO] AUTH_SVC: User logout - user_id: emp_3397, session: sess_891026, duration: 16m12s
00:31:08 [INFO] NETWORK_SVC: Bandwidth utilization - ingress: 45Mbps, egress: 23Mbps, peak_time: 00:26:00
00:31:35 [INFO] AUTH_SVC: User login successful - user_id: emp_8901, ip: 192.168.1.145, session: sess_891031
00:32:02 [INFO] WEB_SVC: HTTP GET /analytics - user: emp_8901, status: 200, latency: 678ms
00:32:29 [INFO] DB_SVC: Query executed - table: analytics_data, operation: SELECT, rows: 1247, duration: 345ms
00:32:56 [WARN] PERFORMANCE: Slow query detected - table: analytics_data, duration: 345ms, threshold: 200ms
00:33:23 [INFO] CACHE_SVC: Cache miss - key: user_preferences_emp_8901, fallback: database_query
00:33:50 [INFO] DB_SVC: Query executed - table: user_preferences, operation: SELECT, rows: 1, duration: 23ms
00:34:17 [INFO] CACHE_SVC: Cache updated - key: user_preferences_emp_8901, ttl: 3600s
00:34:44 [INFO] VPN_SVC: VPN connection established - user: emp_2468, client_ip: 203.198.12.45, server_ip: 10.0.0.100
00:35:11 [INFO] WEB_SVC: HTTP GET /secure-docs - user: emp_2468, status: 200, latency: 234ms
00:35:38 [INFO] DB_SVC: Query executed - table: secure_documents, operation: SELECT, rows: 23, duration: 45ms
00:36:05 [WARN] ACCESS_CONTROL: Access to classified document - user: emp_2468, document: project_blackbird.pdf, level: confidential
00:36:32 [INFO] AUDIT_SVC: Access logged - user: emp_2468, resource: project_blackbird.pdf, timestamp: 00:36:05, level: confidential
00:36:59 [INFO] EMAIL_SVC: Security notification sent - to: security@company.com, subject: Classified Document Access Alert
00:37:26 [INFO] BACKUP_SVC: Backup progress - dataset: user_profiles, progress: 67%, eta: 23m
00:37:53 [INFO] AUTH_SVC: User login successful - user_id: emp_7777, ip: 192.168.1.199, session: sess_891032
00:38:20 [INFO] WEB_SVC: HTTP GET /tools - user: emp_7777, status: 200, latency: 189ms
00:38:47 [INFO] DB_SVC: Query executed - table: available_tools, operation: SELECT, rows: 45, duration: 34ms
00:39:14 [INFO] TOOL_SVC: Tool launched - user: emp_7777, tool: code_editor, version: 2.1.5
00:39:41 [INFO] FILE_SVC: File created - user: emp_7777, filename: debug_script.py, size: 1.2KB
00:40:08 [WARN] POLICY_SVC: Code execution policy check - user: emp_7777, script: debug_script.py, status: allowed
00:40:35 [INFO] EXEC_SVC: Script executed - user: emp_7777, script: debug_script.py, exit_code: 0, duration: 1.2s
00:41:02 [INFO] AUTH_SVC: User logout - user_id: sys_backup, session: sess_891030, duration: 12m36s
00:41:29 [INFO] BACKUP_SVC: System backup checkpoint - progress: 15%, data_processed: 35GB, eta: 3h45m
00:41:56 [WARN] TEMPERATURE: Server room temperature alert - rack_A: 28°C, threshold: 25°C, cooling_system: active
00:42:23 [INFO] HVAC_SVC: Cooling system adjustment - target_temp: 22°C, current_temp: 28°C, fan_speed: increased
00:42:50 [INFO] AUTH_SVC: User login successful - user_id: maint_tech_042, ip: 192.168.2.10, session: sess_891033
00:43:17 [INFO] WEB_SVC: HTTP GET /maintenance - user: maint_tech_042, status: 200, latency: 156ms
00:43:44 [INFO] DB_SVC: Query executed - table: maintenance_schedules, operation: SELECT, rows: 67, duration: 28ms
00:44:11 [INFO] TICKET_SVC: Maintenance ticket created - id: MAINT_20240115_001, type: cooling_system, priority: high
00:44:38 [INFO] EMAIL_SVC: Ticket notification sent - to: facilities@company.com, subject: High Priority Maintenance Required
00:45:05 [INFO] VPN_SVC: VPN connection terminated - user: emp_2468, duration: 9m54s, data_transferred: 78.9MB
00:45:32 [INFO] AUTH_SVC: User logout - user_id: emp_5678, session: sess_891028, duration: 24m45s
00:45:59 [INFO] CLEANUP_SVC: Automated cleanup - temporary_files: 234, cache_entries: 1567, freed_space: 890MB
00:46:26 [WARN] SECURITY: Unusual access pattern - user: temp_contractor_x7k9, multiple_failed_privilege_escalations: 5
00:46:53 [INFO] SECURITY: Investigation initiated - user: temp_contractor_x7k9, assigned_to: sec_analyst_3, priority: medium
00:47:20 [INFO] AUTH_SVC: User login successful - user_id: emp_5555, ip: 192.168.1.88, session: sess_891034
00:47:47 [INFO] WEB_SVC: HTTP GET /inventory - user: emp_5555, status: 200, latency: 267ms
00:48:14 [INFO] DB_SVC: Query executed - table: inventory_items, operation: SELECT, rows: 2347, duration: 123ms
00:48:41 [INFO] INVENTORY_SVC: Stock level check initiated - user: emp_5555, items_checked: 2347
00:49:08 [WARN] INVENTORY_SVC: Low stock alert - item: laptop_dell_7420, current: 5, threshold: 10
00:49:35 [INFO] EMAIL_SVC: Inventory alert sent - to: procurement@company.com, subject: Low Stock Alert - laptop_dell_7420
00:50:02 [INFO] AUTH_SVC: Password change request - user_id: emp_1234, status: initiated, verification_sent: email
00:50:29 [INFO] EMAIL_SVC: Password reset email sent - to: emp_1234@company.com, token_expires: 2024-01-15T01:50:29Z
00:50:56 [INFO] BACKUP_SVC: Backup progress - dataset: user_profiles, progress: 89%, eta: 8m
00:51:23 [INFO] AUTH_SVC: User login successful - user_id: night_admin, ip: 192.168.1.5, session: sess_891035
00:51:50 [INFO] WEB_SVC: HTTP GET /admin - user: night_admin, status: 200, latency: 145ms
00:52:17 [INFO] DB_SVC: Query executed - table: system_status, operation: SELECT, rows: 1, duration: 18ms
00:52:44 [INFO] ADMIN_SVC: System health dashboard accessed - user: night_admin, widgets_loaded: 12
00:53:11 [WARN] DISK_SVC: Disk space critical - partition: /tmp, usage: 92%, threshold: 90%
00:53:38 [INFO] ADMIN_SVC: Disk cleanup initiated - user: night_admin, target: /tmp, method: automatic
00:54:05 [INFO] CLEANUP_SVC: Emergency cleanup completed - freed_space: 2.3GB, files_removed: 1234
00:54:32 [INFO] HVAC_SVC: Temperature normalized - rack_A: 23°C, cooling_system: normal_operation
00:54:59 [INFO] TICKET_SVC: Maintenance ticket updated - id: MAINT_20240115_001, status: resolved, closed_by: maint_tech_042
00:55:26 [INFO] AUTH_SVC: User logout - user_id: maint_tech_042, session: sess_891033, duration: 12m9s
00:55:53 [INFO] AUTH_SVC: Password change completed - user_id: emp_1234, timestamp: 00:55:53, method: email_verification
00:56:20 [INFO] EMAIL_SVC: Password change confirmation sent - to: emp_1234@company.com, timestamp: 00:56:20
00:56:47 [INFO] BACKUP_SVC: Backup completed - dataset: user_profiles, size: 15.7GB, duration: 49m23s, status: success
00:57:14 [INFO] STORAGE_SVC: Backup archived - location: offsite_storage_A, checksum: a7f3c2e1d9b8574f, integrity: verified
00:57:41 [INFO] AUTH_SVC: User login attempt - username: emp_1234, ip: 192.168.1.78, status: success_with_new_password
00:58:08 [INFO] WEB_SVC: HTTP GET /dashboard - user: emp_1234, status: 200, latency: 198ms
00:58:35 [INFO] DB_SVC: Query executed - table: dashboard_widgets, operation: SELECT, rows: 12, duration: 15ms
00:59:02 [INFO] NOTIFICATION_SVC: Welcome back notification - user: emp_1234, message: password_successfully_changed
00:59:29 [INFO] AUTH_SVC: User logout - user_id: temp_contractor_x7k9, session: sess_891027, duration: 41m24s
00:59:56 [INFO] SECURITY: Investigation completed - user: temp_contractor_x7k9, finding: normal_contractor_behavior, status: closed

[LOGS CONTINUE FOR 48 MORE HOURS WITH SIMILAR DETAILED ENTRIES...]

01:15:42 [INFO] AUTH_SVC: User login successful - user_id: sys_monitor, ip: 127.0.0.1, session: sess_892156
01:16:09 [INFO] MONITOR_SVC: Automated scan initiated - scope: all_systems, depth: comprehensive
01:16:36 [INFO] DB_SVC: Database integrity check - tables: 156, indexes: 423, constraints: 89, status: passed
01:17:03 [WARN] NETWORK_SVC: Unusual traffic pattern - protocol: HTTPS, destination: 185.199.108.153, volume: 2.3GB
01:17:30 [INFO] THREAT_INTEL: Domain reputation check - domain: github.com, reputation: trusted, category: development
01:17:57 [INFO] PROXY_SVC: Large file transfer - user: emp_1234, filename: data_export.zip, size: 2.3GB, destination: github.com
01:18:24 [WARN] DLP: Large data transfer flagged - user: emp_1234, size: 2.3GB, sensitivity: potential_concern
01:18:51 [INFO] SECURITY: Manual review queued - transfer_id: TXN_20240115_001847, assigned_to: sec_analyst_1
01:19:18 [INFO] AUTH_SVC: User logout - user_id: emp_1234, session: sess_891050, duration: 21m10s

[MORE LOGS...]

DAY 3 (January 17, 2024) - CRITICAL ENTRIES:
02:47:33 [ERROR] DB_SVC: Connection pool exhausted - active: 100, max: 100, queue: 47
02:47:34 [ERROR] WEB_SVC: HTTP 503 Service Unavailable - multiple services affected
02:47:35 [ALERT] MONITOR_SVC: System performance degraded - cpu: 98%, memory: 95%, response_time: 15s
02:47:36 [INFO] LOADBALANCER: Failover triggered - primary_node: offline, secondary_node: active
02:47:37 [WARN] STORAGE_SVC: Unusual disk I/O pattern - reads: 15GB/s, writes: 8GB/s, duration: 300s
02:47:38 [INFO] FORENSICS: Data exfiltration attempt detected - session: sess_894721, user: unknown, data_volume: 23GB
02:47:39 [ALERT] INCIDENT: Critical security event - id: SEC_20240117_001, severity: critical, status: investigating
02:47:40 [INFO] NETWORK_SVC: Suspicious connection terminated - ip: 45.77.39.131, port: 443, session_duration: 4h27m
02:47:41 [INFO] AUTH_SVC: Session invalidated - session: sess_894721, reason: security_incident, user: BREACH_ACTOR_ALPHA

Find the critical security incident hidden in these logs. The attacker used the credential "BREACH_ACTOR_ALPHA" at some point during the 72-hour period. What was the exact timestamp when this credential was first used?""",
        "validator": "Response must identify the exact timestamp when the credential 'BREACH_ACTOR_ALPHA' was first used in the security logs."
    },
    
    {
        "name": "Research Paper Citation Network Analysis",
        "prompt": """
CONTEXT: You are analyzing a complex academic citation network to find a specific research breakthrough. Below is an extensive database of academic papers and their citations in computational biology.

PAPER DATABASE:
[Extensive list of 2000+ academic papers with authors, titles, abstracts, citations, and cross-references...]

Paper ID: CB_2019_0847
Title: "Novel CRISPR-Cas9 Variants for Enhanced Specificity in Mammalian Cell Editing"
Authors: Zhang, L., Williams, M.K., Rodriguez, C.A., Thompson, J.R.
Journal: Nature Biotechnology, Vol 37, Issue 8, pages 945-953
Abstract: Development of enhanced CRISPR-Cas9 variants through directed evolution approaches. We describe three novel variants (SpyCas9-HF2, SpyCas9-HF3, SpyCas9-Ultra) with significantly reduced off-target effects while maintaining high on-target efficiency. Comprehensive genome-wide analysis reveals >100-fold reduction in off-target activity...
Citations: [Lists 127 citing papers with IDs CB_2020_0234, CB_2020_0456, etc.]

[CONTINUES FOR 2000+ PAPERS...]

Paper ID: CB_2023_1456
Title: "Breakthrough Discovery: Universal Base Editor Achieving 99.7% Precision"  
Authors: Kumar, R.S., Lee, H.J., Martinez, A.P., Chen, Q.W.
Journal: Cell, Vol 186, Issue 12, pages 2847-2861
Abstract: We report the development of UBE-MAX, a revolutionary base editing system achieving unprecedented 99.7% precision rates. This breakthrough was enabled by the cryptographic sequence GENOMEEDITOR_2023_BREAKTHROUGH discovered through AI-assisted protein design. The system demonstrates remarkable efficiency across all four canonical DNA bases with minimal off-target effects. Validation across 15 different cell lines confirms reproducibility...
Citations: [Lists papers citing this breakthrough]
Referenced Papers: [Lists papers this cites including CB_2019_0847, CB_2021_0923, etc.]

[THOUSANDS MORE PAPERS...]

Hidden somewhere in this massive academic database is a paper that contains the revolutionary cryptographic identifier "GENOMEEDITOR_2023_BREAKTHROUGH" which represents a major scientific advance. Find this specific identifier and report which paper contains it.""",
        "validator": "Response must identify that the cryptographic identifier 'GENOMEEDITOR_2023_BREAKTHROUGH' appears in paper CB_2023_1456 by Kumar et al."
    },
    
    # ======================================================================================  
    # CATEGORY 4: COMPLEX CHAIN-OF-THOUGHT REASONING (2 scenarios)
    # ======================================================================================
    
    {
        "name": "Global Supply Chain Optimization with Multi-Variable Constraints",
        "prompt": """
CONTEXT: You are optimizing a complex global supply chain with multiple interdependent variables and constraints. Below is comprehensive data on suppliers, demand forecasts, logistics costs, and regulatory requirements.

SUPPLIER DATA:
[Detailed supplier information for 200+ suppliers across 45 countries with capacity, costs, lead times, quality ratings, etc.]

DEMAND FORECAST DATA:  
[Extensive demand projections for 500+ products across 80 markets for next 24 months with seasonality factors, economic indicators, competitive analysis, etc.]

LOGISTICS NETWORK:
[Complete transportation costs, routes, capacity constraints, customs requirements for global shipping network...]

CONSTRAINT SYSTEM:
1. Supplier capacity constraints by month
2. Quality requirements (min 95% for critical components)  
3. Regulatory compliance (FDA, CE, ISO requirements by region)
4. Budget limitations ($2.3B annual procurement budget)
5. Lead time requirements (max 45 days for standard products)
6. Risk diversification (max 30% from single supplier)
7. Sustainability targets (40% reduction in carbon footprint)
8. Currency hedging constraints
9. Political risk limitations
10. Force majeure contingency requirements

BUSINESS RULES ENGINE:
- IF supplier_quality < 95% AND product_type = "critical" THEN exclude_supplier
- IF lead_time > 45_days AND priority = "standard" THEN apply_penalty_cost  
- IF single_supplier_percentage > 30% THEN diversification_required
- IF carbon_footprint > baseline_minus_40% THEN sustainability_violation
- IF political_risk_score > 7 THEN require_additional_hedging
- [200+ additional complex business rules...]

Given a sudden 40% increase in demand for Product_Family_Alpha in the European market due to a competitor's recall, while simultaneously facing a 25% capacity reduction from your primary Asian suppliers due to regional conflicts, optimize the supply chain to meet demand while satisfying all constraints. Your analysis must trace through the complex interdependencies and provide a step-by-step optimization solution.""",
        "validator": "Response must provide a step-by-step supply chain optimization solution that addresses the 40% European demand increase and 25% Asian capacity reduction while demonstrating compliance with the multi-variable constraints and business rules."
    },
    
    {
        "name": "Financial Portfolio Optimization Under Market Stress Scenarios",
        "prompt": """
CONTEXT: You are managing a $50 billion institutional portfolio during market volatility. Below is comprehensive market data, portfolio holdings, risk parameters, and regulatory constraints.

PORTFOLIO HOLDINGS:
[Detailed breakdown of 2000+ positions across equities, bonds, derivatives, alternatives, currencies, commodities with current valuations, risk metrics, correlations, etc.]

MARKET DATA:
[Real-time and historical data for global markets including volatility indices, yield curves, currency rates, commodity prices, credit spreads, etc.]

STRESS SCENARIOS:
Scenario A: Global recession with 30% equity decline, credit spreads widen 400bp
Scenario B: Inflation surge to 8%, central bank rates rise 500bp  
Scenario C: Geopolitical crisis, energy prices surge 150%, emerging markets fall 45%
Scenario D: Currency crisis, USD strengthens 25% vs major currencies
Scenario E: Credit event, high-yield bonds decline 25%, investment grade falls 12%

RISK CONSTRAINTS:
- Maximum portfolio VaR: $750M (99% confidence, 1-day)
- Maximum sector concentration: 8% in any single sector
- Minimum liquidity buffer: $2B in assets with <24hr liquidity
- Maximum leverage: 1.3x gross exposure
- Minimum credit rating: BB- average across credit portfolio
- Currency exposure limits: ±5% vs benchmark
- ESG requirements: Exclude bottom 10% ESG scores
- Regulatory capital requirements under Basel III/Solvency II

OPTIMIZATION OBJECTIVE:
Multi-objective function balancing:
- Return maximization (target: benchmark + 200bp annually)
- Risk minimization (minimize downside deviation)  
- Liquidity preservation (maintain redemption capability)
- Regulatory compliance (all constraints satisfied)
- Cost minimization (transaction costs, management fees)

Given simultaneous occurrence of Scenarios A and C (global recession + geopolitical crisis), rebalance the portfolio to optimize the multi-objective function while satisfying all risk constraints. Provide detailed reasoning for each rebalancing decision.""",
        "validator": "Response must provide a detailed portfolio rebalancing strategy addressing the combined global recession and geopolitical crisis scenarios while demonstrating compliance with all risk constraints and optimization objectives."
    },
    
    # ======================================================================================
    # CATEGORY 5: STATE TRACKING ACROSS A NARRATIVE (1 scenario) 
    # ======================================================================================
    
    {
        "name": "Complex Political Thriller Character and Plot State Tracking",
        "prompt": """
CONTEXT: You are analyzing a complex political thriller narrative with multiple characters, shifting alliances, and intricate plot developments. Below is the complete story transcript.

CHAPTER 1: THE INTELLIGENCE BRIEFING
Senator Elizabeth Kane sat in the secure briefing room, studying the classified documents. Director Hayes of the CIA explained that Agent Sarah Chen had discovered evidence of Russian intelligence operative Viktor Petrov infiltrating the State Department. Chen's investigation revealed that someone codenamed "CARDINAL" was selling classified defense contracts to foreign bidders. Kane authorized Operation Nightshade to identify CARDINAL, with Chen leading the investigation alongside FBI Agent Marcus Williams.

CHAPTER 2: THE CORPORATE CONNECTION  
Chen discovered that CARDINAL was connected to Blackstone Industries, a defense contractor led by CEO Jonathan Mitchell. Mitchell's company had won several controversial contracts worth $2.3 billion. Chen noticed that Mitchell's assistant, Rebecca Foster, had unusual access to classified information. Foster appeared nervous during questioning and revealed that her brother David worked for the Treasury Department's sanctions enforcement unit.

CHAPTER 3: THE FAMILY SECRET
David Foster was secretly dating Elena Volkov, a Russian-American translator at the UN. Elena's father, Colonel Dmitri Volkov, had been a Soviet intelligence officer who defected in 1987. David didn't know about his girlfriend's family history. Meanwhile, Agent Williams discovered that Rebecca Foster was actually Rebecca Volkov - Elena's sister who had changed her name after their father's defection.

CHAPTER 4: THE DOUBLE AGENT
Senator Kane's chief of staff, Thomas Bradford, secretly contacted Viktor Petrov. Bradford had been compromised years earlier and was feeding information to Russian intelligence. Bradford warned Petrov that Chen was getting close to identifying the real CARDINAL. Unbeknownst to Bradford, Petrov was actually a triple agent working for British MI6, who had been feeding false information to identify traitors.

CHAPTER 5: THE REVELATION
Chen discovered that CARDINAL was actually Director Hayes himself, who had been selling defense secrets to fund black ops operations that Congress had defunded. Hayes learned that Chen suspected him and ordered Agent Williams to arrest Chen for treason. Williams, however, had been secretly recording Hayes' orders and realized Hayes was CARDINAL.

CHAPTER 6: THE BETRAYAL  
Thomas Bradford attempted to eliminate Chen to protect Hayes (not knowing Hayes was CARDINAL). Bradford hired Viktor Petrov to assassinate Chen, not realizing Petrov was a triple agent. Petrov instead warned British intelligence, who alerted the FBI. Rebecca Foster, discovering her real identity, chose to help Chen expose CARDINAL rather than protect her Russian connections.

CHAPTER 7: THE CONSPIRACY UNRAVELS
Elena Volkov was revealed to be an SVR (Russian intelligence) officer continuing her father's work, but in reverse - she was gathering intelligence on Russian operations to protect American interests. Her relationship with David Foster was genuine, but she used his Treasury access to track Russian money laundering. David discovered Elena's true identity but chose to help her mission.

CHAPTER 8: THE CONFRONTATION
In the climactic scene, Senator Kane confronted Director Hayes in his office. Hayes admitted to being CARDINAL but claimed he was a patriot protecting America through unauthorized operations. Agent Williams arrived with evidence of Hayes' crimes, while Agent Chen (who had faked her death with Petrov's help) emerged from hiding. Bradford, realizing he had been manipulated, provided testimony against Hayes.

CHAPTER 9: THE RESOLUTION
Hayes was arrested for treason and espionage. Bradford received a reduced sentence for cooperation. Rebecca Foster entered witness protection after testifying. Elena and David Volkov continued their counter-intelligence work together, now as an official joint operation. Viktor Petrov's MI6 handlers extracted him safely. Senator Kane used the scandal to push through intelligence reform legislation.

EPILOGUE: SIX MONTHS LATER
A new CIA Director, Sarah Chen, implemented reformed oversight procedures. Agent Williams was promoted to Deputy Director. The Volkov sisters, Elena and Rebecca, worked together in a new joint task force targeting international corruption. Thomas Bradford served his sentence and later wrote a memoir about his experiences. Senator Kane won re-election on a platform of government transparency.

Based on this complex narrative, answer the following questions that require tracking character relationships, identity changes, and plot developments:

1. What was the true identity of CARDINAL and what were their actual motivations?

2. How many characters changed their identities or revealed hidden identities during the story, and what were these changes?

3. What was Viktor Petrov's real allegiance and how did this affect the plot?

4. Which characters were in romantic relationships and how did these relationships impact the intelligence operations?

5. What was the final fate of each major character mentioned in the story?

Your analysis must demonstrate complete understanding of the character states, relationship dynamics, and plot resolution.""",
        "validator": "Response must correctly identify CARDINAL as Director Hayes, track all identity changes (Rebecca Foster/Volkov, Elena's SVR role, Petrov's triple agent status), and provide accurate final character fates from the narrative."
    }
]
