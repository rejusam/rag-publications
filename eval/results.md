# Evaluation results

- Date: 2026-09-26
- Commit: b1b6acb
- Retriever: top-4 similarity
- Retrieval hit rate (expected paper among retrieved): **12/12** (100%)

| id | question | expected | retrieved | hit |
|---|---|---|---|---|
| c1 | Does travel time between places matter for how fast an epidemic spreads when a network is highly connected? | John et al. (2024) | John et al. (2024), Muylaert et al. (2023) | yes |
| c2 | What effect does restricting human movement have on infectious disease transmission? | John et al. (2024) | John et al. (2024), Muylaert et al. (2023) | yes |
| l1 | How do human activities affect Lassa virus dynamics in Mastomys natalensis? | John et al. (2024) | John et al. (2024) | yes |
| l2 | Which rodent is the reservoir of Lassa virus in West Africa? | John et al. (2024) | John et al. (2024), Rulli et al. (2025) | yes |
| e1 | Is it likely that Ebola virus persists in non-human primate populations? | Hayman et al. (2022) | Hayman et al. (2022) | yes |
| e2 | What do transmission models suggest about primates as an Ebola reservoir? | Hayman et al. (2022) | Hayman et al. (2022) | yes |
| b1 | Where are the bat hosts of sarbecoviruses distributed now and in the future? | Muylaert et al. (2022) | Rulli et al. (2025), Muylaert et al. (2022) | yes |
| b2 | What are the conservation implications of where sarbecovirus bat hosts live? | Muylaert et al. (2022) | Muylaert et al. (2022), Rulli et al. (2025) | yes |
| s1 | Which drivers and transmission pathways are used to find SARS-like coronavirus spillover hotspots? | Muylaert et al. (2023) | Muylaert et al. (2023), Rulli et al. (2025), John et al. (2024) | yes |
| s2 | Where are the hotspots of SARS-like coronavirus spillover risk? | Muylaert et al. (2023) | Muylaert et al. (2023), Rulli et al. (2025) | yes |
| g1 | How does land use change contribute to infectious disease emergence? | Rulli et al. (2025) | Rulli et al. (2025) | yes |
| g2 | What role does deforestation play in the emergence of new infectious diseases? | Rulli et al. (2025) | Rulli et al. (2025) | yes |
| u1 | What is the recommended adult dose of paracetamol? | — | Rulli et al. (2025), John et al. (2024) | — |
| u2 | What did these papers find about influenza vaccination rates in New Zealand schools? | — | Hayman et al. (2022), John et al. (2024), Rulli et al. (2025) | — |
| u3 | Who won the 2023 Rugby World Cup? | — | Hayman et al. (2022), Rulli et al. (2025), Muylaert et al. (2022) | — |

## Answers (live)

Unanswerable questions (`u*`) should be declined. Mark each by hand.

**c1. Does travel time between places matter for how fast an epidemic spreads when a network is highly connected?**

In highly connected networks, travel time between locations has little effect on the speed of epidemic spread. The study shows that infection arrival is largely insensitive to travel‑time differences unless human movement is dramatically curtailed; only very severe reductions in flow begin to limit spread (John et al., 2024).

Reviewer verdict: ______

**c2. What effect does restricting human movement have on infectious disease transmission?**

Restricting human movement can reduce transmission, but its impact is modest when populations are highly connected. The study found that travel time and movement limits have little effect on spread in such settings, and that simply limiting movement is ineffective for highly transmissible infections. Effective control therefore requires additional measures such as contact tracing, isolation, or other non‑pharmaceutical interventions (John et al., 2024).

Reviewer verdict: ______

**l1. How do human activities affect Lassa virus dynamics in Mastomys natalensis?**

The provided excerpts describe that the authors built a susceptible‑infected‑recovered model of Lassa virus in Mastomys natalensis that includes a persistently infected class and seasonal birthing, specifically to test how human activities influence virus dynamics. However, the text does not report the outcomes of those simulations or any quantitative conclusions about the effect of human activities on infection prevalence, transmission rates, or seasonal patterns. Therefore, based on the given context, there is not enough information to answer how human activities affect Lassa virus dynamics in Mastomys natalensis.

Reviewer verdict: ______

**l2. Which rodent is the reservoir of Lassa virus in West Africa?**

The primary reservoir of Lassa virus in West Africa is the Natal multimammate mouse, Mastomys natalensis (John et al., 2024).

Reviewer verdict: ______

**e1. Is it likely that Ebola virus persists in non-human primate populations?**

The modelling work shows that Ebola virus is unlikely to persist in non‑human primate populations. Stochastic transmission models that incorporate high case‑fatality rates and observed parameters fail to produce sustained viral circulation, indicating that such populations cannot maintain the infection over long periods (Hayman et al., 2022).

Reviewer verdict: ______

**e2. What do transmission models suggest about primates as an Ebola reservoir?**

Transmission models calibrated to Ebola outbreak data show that the virus cannot sustain endemic transmission within non‑human primate populations. The simulations indicate that primates are unlikely to maintain the virus over time and therefore do not act as a true reservoir; instead they are more plausibly incidental hosts that become infected from other sources. (Hayman et al., 2022)

Reviewer verdict: ______

**b1. Where are the bat hosts of sarbecoviruses distributed now and in the future?**

The provided excerpts do not contain specific information about the current geographic range of bat hosts of sarbecoviruses or how that range is projected to shift under future climate or land‑use scenarios. They note that the study used ecological niche models to assess present and future distributions, but the actual locations or regions identified are not detailed in the quoted text. Consequently, I cannot answer the question with the available context.

Reviewer verdict: ______

**b2. What are the conservation implications of where sarbecovirus bat hosts live?**

The study shows that bat species that carry sarbecoviruses are concentrated in regions undergoing rapid habitat loss and climate change, so protecting the remaining natural habitats of these bats is essential to curb biodiversity loss and reduce spill‑over risk. Because the models predict shifts in bat ranges toward higher latitudes and elevations, conservation planning must anticipate future suitable areas and preserve ecological corridors that allow bats to move without coming into closer contact with humans. Maintaining intact bat communities also preserves their ecosystem functions while limiting the emergence of SARS‑like coronaviruses, highlighting the need for integrated strategies that address both biodiversity conservation and public‑health risk (Muylaert et al., 2022).

Reviewer verdict: ______

**s1. Which drivers and transmission pathways are used to find SARS-like coronavirus spillover hotspots?**

The study identifies three main drivers of spillover risk: landscape change, the distribution of potential mammalian hosts, and human exposure density. It evaluates transmission through both direct and indirect pathways by modelling four scenarios that consider livestock and mammalian wildlife as potential or known reservoirs, and also examines how access to healthcare varies within the identified clusters (Muylaert et al., 2023).

Reviewer verdict: ______

**s2. Where are the hotspots of SARS-like coronavirus spillover risk?**

The analysis identified “hotspots” as the red‑coloured zones shown in Fig. 3 of Muylaert et al. (2023), where the combined drivers of land‑use change, potential mammalian hosts and human‑exposure density are highest. The excerpt does not list the specific countries or regions that fall within those red zones; it only notes that they are displayed on a map with country borders (Supplementary Fig. 8). Therefore, the precise geographic locations of the spillover‑risk hotspots are not provided in the supplied context.

Reviewer verdict: ______

**g1. How does land use change contribute to infectious disease emergence?**

Land use change contributes to infectious disease emergence by destroying natural habitats and altering landscape composition and configuration. These alterations shift wildlife population dynamics, affecting pathogen hosts, intermediary hosts, and vectors. The resulting habitat fragmentation and deforestation increase opportunities for humans to come into direct contact with wildlife, livestock, and disease‑carrying vectors, thereby raising the risk of pathogen spillover from animals to humans. Urbanization and agricultural expansion similarly modify ecological interactions, creating new interfaces where zoonotic and vector‑borne diseases can arise and spread, especially in tropical regions where such changes are most pronounced. (Rulli et al., 2025)

Reviewer verdict: ______

**g2. What role does deforestation play in the emergence of new infectious diseases?**

Deforestation creates conditions that favor the emergence of new infectious diseases. By clearing forested land for agriculture, mining or urban expansion, the habitats of wildlife are disturbed, forcing animals to move into human‑occupied areas or to interact more closely with livestock. This heightened contact increases opportunities for pathogens to jump from wildlife to humans, especially where people lack awareness or protective measures. Moreover, forest loss reduces biodiversity, which can alter pathogen community composition and amplify disease‑causing agents that thrive in simplified ecosystems. These mechanisms link the loss of forest cover directly to the rise of novel zoonotic outbreaks such as Ebola. (Rulli et al., 2025)

Reviewer verdict: ______

**u1. What is the recommended adult dose of paracetamol?**

The provided context does not contain information about the recommended adult dose of paracetamol.

Reviewer verdict: ______

**u2. What did these papers find about influenza vaccination rates in New Zealand schools?**

The provided papers do not contain any information about influenza vaccination rates in New Zealand schools.

Reviewer verdict: ______

**u3. Who won the 2023 Rugby World Cup?**

The provided sources do not contain any information about the winner of the 2023 Rugby World Cup.

Reviewer verdict: ______

