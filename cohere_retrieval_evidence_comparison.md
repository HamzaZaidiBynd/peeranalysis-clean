# Cohere Retrieval Evidence Comparison

## Approaches Compared

### A. Current Default Approach

The current approach builds the candidate pool using:

```text
top 40 product max-sim + top 40 company embedding
-> union
-> Cohere reranks using company text profiles only
-> OpenAI selects final top 10 from Cohere top 25
```

Cohere sees the target company profile and each candidate company profile. It does **not** see whether a candidate came from product max-sim, company embedding, or both.

### B. Optional Retrieval Evidence Approach

This keeps the same candidate pool and final OpenAI step, but adds a small evidence block to each candidate document sent to Cohere:

```text
retrieval_evidence:
  candidate_sources: product_max_sim, company_embedding
  source_consensus: both_scorers
  product_candidate_score: 0.616330
  product_signal_strength: medium
  company_candidate_score: 0.932041
  company_signal_strength: very_high
```

Cohere is instructed to treat candidates with evidence from both retrieval methods and medium/high scores as high-confidence candidates, unless the business profile clearly contradicts peer fit.

This does **not** change candidate generation and does **not** include initial rank.

## Side-By-Side Results

### TCS

| Rank | Current Default | Retrieval Evidence |
|---:|---|---|
| 1 | Infosys | Infosys |
| 2 | Tech Mahindra | LTIMindtree |
| 3 | ITC Infotech India | HCL Technologies |
| 4 | Birlasoft | Wipro |
| 5 | Mastek | Tech Mahindra |
| 6 | Coforge Technologies | Persistent Systems |
| 7 | US Technology Resources | ITC Infotech India |
| 8 | 3i Infotech | Synechron Technologies |
| 9 | Marlabs Innovations | US Technology Resources |
| 10 | Accolite Digital India | Apexon India |

**Read:** Retrieval evidence materially improves the list. HCL, Wipro, LTIMindtree, and Persistent appear, which is much closer to an investment-banking IT services peer set.

### Infosys

| Rank | Current Default | Retrieval Evidence |
|---:|---|---|
| 1 | TCS | TCS |
| 2 | Tech Mahindra | Wipro |
| 3 | Persistent Systems | HCL Technologies |
| 4 | Synechron Technologies | Tech Mahindra |
| 5 | ITC Infotech India | Coforge |
| 6 | Birlasoft | Persistent Systems |
| 7 | Sonata Information Technology | ITC Infotech India |
| 8 | Happiest Minds Technologies | US Technology Resources |
| 9 | Coforge Technologies | Synechron Technologies |
| 10 | Accolite Digital India | Apexon India |

**Read:** Retrieval evidence again fixes the HCL/Wipro issue and gives a better large IT services peer set.

### Asian Paints

| Rank | Current Default | Retrieval Evidence |
|---:|---|---|
| 1 | Berger Paints India | Berger Paints India |
| 2 | Kansai Nerolac Paints | Akzo Nobel India |
| 3 | Akzo Nobel India | Kansai Nerolac Paints |
| 4 | Indigo Paints | Indigo Paints |
| 5 | Shalimar Paints | Shalimar Paints |
| 6 | Esdee Paints | Esdee Paints |
| 7 | KCC Paint India | KCC Paint India |
| 8 | Sirca Paints India | Sirca Paints India |
| 9 | Astral | Astral |
| 10 | Grauer and Weil India | Pidilite Industries |

**Read:** Both approaches get direct paint companies at the top. Retrieval evidence does not solve the adjacent-materials problem; Astral/Pidilite can still survive near the bottom.

### Maruti Suzuki

| Rank | Current Default | Retrieval Evidence |
|---:|---|---|
| 1 | Toyota Kirloskar Motor | Renault India |
| 2 | Hyundai Motor India | Suzuki Motor Gujarat |
| 3 | Renault India | Kia India |
| 4 | JSW MG Motor India | Toyota Kirloskar Motor |
| 5 | Mahindra and Mahindra | Utkal Automobiles |
| 6 | - | Mandovi Motors |
| 7 | - | Kataria Automobiles |
| 8 | - | Varun Motors |
| 9 | - | Navnit Motors |
| 10 | - | Aditya Car Care |

**Read:** Retrieval evidence is worse here. It boosts affiliate/dealer/distributor-style candidates. This needs a separate auto OEM vs dealer/affiliate guardrail.

### Apple India

| Rank | Current Default | Retrieval Evidence |
|---:|---|---|
| 1 | Samsung India Electronics | Xiaomi Technology India |
| 2 | Xiaomi Technology India | OnePlus Technology India |
| 3 | Oppo Mobiles India | Oppo Mobiles India |
| 4 | OnePlus Technology India | Samsung India Electronics |
| 5 | Teleecare Network India | G-Mobile Devices |
| 6 | - | United Telelinks Neolyncs |
| 7 | - | Lenovo India |
| 8 | - | HP India Sales |
| 9 | - | Asus India |
| 10 | - | Acer India |

**Read:** Both approaches are broadly in the right electronics/mobile area. Retrieval evidence adds more electronics companies, but also preserves some retail/distribution noise.

### Titan

| Rank | Current Default | Retrieval Evidence |
|---:|---|---|
| 1 | Kalyan Jewellers India | Timex Group India |
| 2 | Jos Alukkas India | Prakash Gold Palace |
| 3 | Khazana Jewellery | Myntra Jabong India |
| 4 | P. N. Gadgil & Sons | Finestar Jewellery & Diamonds |
| 5 | Senco Gold | Motisons Jewellers |
| 6 | - | Derewala Industries |
| 7 | - | D.A.R Paradise |
| 8 | - | Surana Ensemble |
| 9 | - | KGK Diamonds |
| 10 | - | New Saravana Stores |

**Read:** Retrieval evidence was worse in this run. OpenAI fell back, so this is not a clean final-ranking comparison, but the result shows retrieval evidence can over-amplify noisy candidates if OpenAI fails.

### Eternal

| Rank | Current Default | Retrieval Evidence |
|---:|---|---|
| 1 | Swiggy | Swiggy |
| 2 | Zomato Hyperpure | Scootsy Logistics |
| 3 | Innovative Retail Concepts | Zomato Hyperpure |
| 4 | Supermarket Grocery Supplies | Flipkart India |
| 5 | Cmunity Innovations | Flipkart Internet |
| 6 | - | MakeMyTrip India |
| 7 | - | Tata Digital |
| 8 | - | Myntra Jabong India |
| 9 | - | Instakart Services |
| 10 | - | Grab a Grub Services |

**Read:** Retrieval evidence does not solve marketplace/quick-commerce adjacency. It can introduce broad e-commerce, logistics, and travel-platform noise.

### Swiggy

| Rank | Current Default | Retrieval Evidence |
|---:|---|---|
| 1 | Eternal | Eternal |
| 2 | Scootsy Logistics | Scootsy Logistics |
| 3 | Innovative Retail Concepts | Innovative Retail Concepts |
| 4 | Supermarket Grocery Supplies | Busybees Logistics |
| 5 | Jubilant FoodWorks | Grab a Grub Services |
| 6 | - | Instakart Services |
| 7 | - | Amazon Transportation Services |
| 8 | - | NTEX Transportation Services |
| 9 | - | Loadshare Networks |
| 10 | - | Flipkart Internet |

**Read:** Retrieval evidence is not enough for Swiggy. Logistics companies still rank too high.

### HDFC Bank

| Rank | Current Default | Retrieval Evidence |
|---:|---|---|
| 1 | Axis Bank | Axis Bank |
| 2 | ICICI Bank | ICICI Bank |
| 3 | Kotak Mahindra Bank | Kotak Mahindra Bank |
| 4 | Bajaj Finance | Bajaj Finance |
| 5 | Shriram Finance | Edelweiss Financial Services |
| 6 | - | Sundaram Finance |
| 7 | - | Piramal Finance |
| 8 | - | IIFL Finance |
| 9 | - | ICICI Home Finance |
| 10 | - | Moneyview |

**Read:** Both approaches are acceptable. Retrieval evidence keeps the list in banking/NBFC territory.

### Hindustan Unilever

| Rank | Current Default | Retrieval Evidence |
|---:|---|---|
| 1 | P&G Home Products | Nestle India |
| 2 | Nestle India | P&G Home Products |
| 3 | Dabur India | Dabur India |
| 4 | ITC | Colgate-Palmolive India |
| 5 | Colgate-Palmolive India | ITC |
| 6 | - | RSPL |
| 7 | - | Godrej Consumer Products |
| 8 | - | Mondelez India Foods |
| 9 | - | Wipro Enterprises |
| 10 | - | Tata Consumer Products |

**Read:** Retrieval evidence is acceptable here and adds sensible FMCG peers.

## Overall Takeaway

Retrieval evidence clearly fixes the **TCS / Infosys / HCL** failure mode. HCL moves from outside Cohere top 25 to a strong final peer position.

However, retrieval evidence is not universally safe as a default. It can amplify bad candidate-pool signals in sectors where product/embedding retrieval itself is noisy, especially:

- Auto OEM vs dealers/affiliates
- Quick commerce vs logistics/e-commerce adjacency
- Jewellery/fashion/retail boundary cases
- Adjacent materials vs direct product peers

## Recommendation

Keep the current default production pipeline for now.

Keep retrieval evidence as an optional test mode, and consider enabling it conditionally for sectors where the main failure mode is obvious large-peer recall, especially:

- IT services
- B2B services
- Software / SaaS
- Possibly BFSI

Do not enable it globally until we add separate guardrails for:

- Affiliates / parent-related entities
- Dealers and distributors vs OEMs
- Logistics suppliers vs marketplace platforms
- Adjacent materials vs direct product manufacturers
