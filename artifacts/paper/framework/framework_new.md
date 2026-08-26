## Framework

The research question behind this work is: how can the structural and
analytical reasoning in a scientific data analysis be made explicit,
repeatable, and independently inspectable?

Scientific analysis often depends on decisions that remain scattered across
prose, notebooks, and one-off code: what each field represents, which
observations belong together, what should be measured, and which checks should
pass. When those decisions are not recorded as an executable specification,
each new record, collaborator, or implementation may reconstruct them
differently.

An expert typically performs a structured analysis before time-series observations can support further reasoning. They define relevant conditions, identify boundaries, group observations into occurrences, and calculate properties of those occurrences. Much of this work is deterministic, but it is often repeated by every person or computational system that receives the raw data.

FeatureGraph separates that deterministic work from downstream interpretation. It converts an operational definition of behavior into explicit states, events, bounded occurrences, measurements, and relationships. The resulting records can then be handed to an analyst, database, or other computational system without requiring that consumer to reconstruct the same analysis from the original observations. Scientific meaning and conclusions remain the responsibility of researchers and domain experts.
