## Framework

The research question behind this analysis is: "If I lost access to LLMs tomorrow, what functionality would I lose and what would I be able to retain?"

When an LLM carries out an analysis, much of that analysis can be lost and need to be reconstructed again. Because the process is stochastic, the results may differ. A researcher who could preserve the results of an LLM analysis and carry it out repeatedly in an automated way but independent of the LLM would have extracted an analysis artifact that they could use repeatedly and share with others. 

An expert typically performs a structured analysis before time-series observations can support further reasoning. They define relevant conditions, identify boundaries, group observations into occurrences, and calculate properties of those occurrences. Much of this work is deterministic, but it is often repeated by every person or computational system that receives the raw data.

FeatureGraph separates that deterministic work from downstream interpretation. It converts an operational definition of behavior into explicit states, events, bounded objects, measurements, and relationships. The resulting behavioral records can then be handed to an analyst, language model, database, or other computational consumer without requiring that consumer to reconstruct the same analysis from the original observations.

