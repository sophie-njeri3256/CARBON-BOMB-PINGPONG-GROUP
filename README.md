Carbon Bombs: Potential Exposure to Climatic Extremes Induced by the Largest Fossil Fuel Projects



Environmental Programming – Assignment 2 · Vrije Universiteit Brussel / IUPWARE · 2026–2027 Supervisors: Prof. Dr. Ir. Elga Salvadore, Amaury Laridon Author(s): JULIA REHIARA, SOPHIE GITAU AND CLAIRE GRANDADAM

Purpose of this notebook



Carbon bombs are the 425 largest fossil fuel extraction projects in the world, each able to emit at least 1 GtCO if its reserves are fully extracted and burned \[1]. The 2025 update of the database \[2] gives a combined potential of about 912 GtCO, more than seven times the remaining 1.5 °C carbon budget (130 GtCO). The largest single project, the Permian Delaware Tight shale oil and gas field (Texas, USA), still contains about 23 GtCO.



This notebook is a step-by-step tutorial that:



&#x20;   Pre-processes the carbon bomb database. It combines the coal and oil \& gas sheets of the 2025 update \[2], links each project to its ID and "new project" status from Kühne et al. \[1], and stores the result as an xarray.Dataset (ds\_carbon\_bombs).

&#x20;   Visualises the carbon bombs. It shows the 20 largest projects with their uncertainty, plus the distribution by country and fuel type.

&#x20;   Analyses the additional lifetime exposure to climate extremes caused by the CO emissions of the Permian Delaware Tight project. The hazards are heatwaves, droughts, crop failures, river floods, wildfires and tropical cyclones. The analysis covers 183 countries and 9 world regions, for birth cohorts from 1950 to 2025. It uses the framework of Laridon et al. \[2], which builds on the lifetime exposure approach of Grant et al. \[5].

&#x20;   Highlights the intergenerational inequity of this exposure: people born today will face far more additional extremes from the same emissions than older generations.



All functions defined here are also collected in scripts/carbon\_bombs\_analysis.py.

References



\[1] Kühne, K., Bartsch, N., Driskell Tate, R., Higson, J., \& Habet, A. (2022). "Carbon Bombs" – Mapping key fossil fuel projects. Energy Policy, 166, 112950. \[2] Laridon, A., Pietroiusti, R., Smith, C., et al. (2026, in prep.). A framework linking project-level greenhouse gas emissions to additional climate extremes exposure and multi-century committed change. \[3] Global Carbon Budget 2025. Earth System Science Data, 18(5). \[4] Abram, N. J., et al. (2025). Quantifying the regional to global climate impacts of individual fossil fuel projects to inform decision-making. npj Climate Action, 4(1), 92. \[5] Grant, L., Vanderkelen, I., Gudmundsson, L., Fischer, E., Seneviratne, S. I., \& Thiery, W. (2025). Global emergence of unprecedented lifetime exposure to climate extremes. Nature, 641, 374–379.

