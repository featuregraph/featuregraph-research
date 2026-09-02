# FeatureGraph Research

FeatureGraph is an open framework for deterministic, inspectable, and
reproducible scientific data analysis.

A researcher specifies what the data contains and what should be grouped,
measured, compared, and checked. FeatureGraph carries out those declared steps
deterministically, returns traceable result tables, and preserves the source
observations behind them. Scientific interpretation remains with researchers
and domain experts.

```text
observations
    → states and events
    → object boundaries and identities
    → behavioral object tables
    → computational queries
```

This repository contains exploratory FeatureGraph studies and research
artifacts. The released software is maintained in
[`featuregraph/featuregraph`](https://github.com/featuregraph/featuregraph).
The current public release is FeatureGraph beta `v0.1.0b1`; it provides
oscillation objects, wave-derived accumulation objects, inspectable
construction features, the validated BIDMC envelope/interval construction,
and detector-discordant handoff records.

> The released beta and the experimental repositories have different version
> contracts. Reproduce released results from the immutable beta tag.

## Origins

FeatureGraph is the current form of a question this work has returned to for a
decade: how do you represent what a system is doing, from ordered observations
of it, in a way another person can check?

The approaches changed. The question did not.

- **2016–2019** — [`smartcab`](https://github.com/habibdraft/smartcab),
  [`rl-agents`](https://github.com/habibdraft/rl-agents): learning agents over
  discrete states and actions.
- **2021** — [`function-approximation`](https://github.com/habibdraft/function-approximation),
  [`markov-processes`](https://github.com/habibdraft/markov-processes):
  approximating continuous behavior, and transitions between states.
- **2021–2022** — [`tmdb`](https://github.com/habibdraft/tmdb),
  [`db-app`](https://github.com/habibdraft/db-app),
  [`bnc`](https://github.com/habibdraft/bnc): pipelines and public APIs — how
  observations arrive, and how they are served.
- **2022** — [`matrix-arithmetic`](https://github.com/habibdraft/matrix-arithmetic):
  interpreting signals sent from an upstream device to a downstream receiver.
- **2024** — [`feature_graph`](https://github.com/habibdraft/feature_graph): the
  first feature graph builder.
- **2026** — [`eventql`](https://github.com/habibdraft/eventql): a declarative
  language for signal transformations over structured numeric data.
- **2026** — FeatureGraph: researcher-declared rules, compiled deterministically
  into inspectable states, events, and bounded objects, with the construction
  recorded alongside the result.

What carried across was the state representation. A reinforcement learning
agent evaluates states, and what it can learn is bounded by how well those
states are described — but the description is usually taken as given, produced
somewhere upstream and rarely examined.

The question that kept returning was what an agent could do with layered,
explicit information about the states it was evaluating, rather than whatever
representation happened to be handed to it. That makes the representation
itself the research problem, and FeatureGraph is the attempt to treat it as
one: states declared rather than inferred, constructed deterministically, and
inspectable independently of whatever consumes them.

Earlier work is at [@habibdraft](https://github.com/habibdraft).

## Install the beta

Install the immutable release:

```bash
python -m pip install "featuregraph @ git+https://github.com/featuregraph/featuregraph.git@v0.1.0b1"
```

For development against the compatible beta maintenance branch:

```bash
git clone --branch beta/v0.1.x --single-branch \
  https://github.com/featuregraph/featuregraph.git
cd featuregraph
python -m pip install -e ".[dev]"
python -m pytest
```

## Minimal example

```python
import featuregraph as fg

bidmc = fg.datasets.bidmc(subject=1)

builder = fg.oscillation.Oscillation(
    signals="respiration",
    group="subject",
)

features = builder.fit_transform(bidmc)
objects = builder.summarize(features, signal="respiration")

long_oscillations = (
    objects.query()
    .where(duration__ge=100)
    .select("oscillation_id", "duration", "amplitude")
    .collect()
)
```

`fit_transform()` retains sample-level observations, states, events, and identities. `summarize()` produces one row per complete oscillation. Queries operate on that explicit representation instead of detecting behavioral boundaries again.

## Documentation and research record

- [Beta documentation](https://featuregraph.readthedocs.io/)
- [Quickstart](https://featuregraph.readthedocs.io/en/latest/quickstart.html)
- [Datasets](https://featuregraph.readthedocs.io/en/latest/datasets.html)
- [API reference](https://featuregraph.readthedocs.io/en/latest/api/index.html)
- [Demonstration notebook](https://github.com/featuregraph/featuregraph/blob/v0.1.0b1/notebooks/demo_notebook.ipynb)
- [Current BIDMC study record](https://github.com/featuregraph/featuregraph/blob/main/artifacts/studies/bidmc_object_workflow_study.md)
- [Current framework paper draft](https://github.com/featuregraph/featuregraph/blob/main/artifacts/paper/master/featuregraph_master_draft.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Release `v0.1.0b1`](https://github.com/featuregraph/featuregraph/releases/tag/v0.1.0b1)
- [Archived research record](https://doi.org/10.5281/zenodo.21984186)
- [Project website](https://featuregraph.ai/?utm_source=github&utm_medium=referral&utm_campaign=featuregraph_research_repository)

## Reproduce the research artifacts

```bash
python scripts/reproduce.py
```

The command reads the versioned reproduction manifest, downloads the fixed BIDMC and Tennessee Eastman selections, reconstructs object tables, generates annotated figures, and records environment and checksum metadata. See the [reproducibility guide](docs/reproducibility.md) for details.

## Living beta research line

The beta maintenance line remains available for asking how well the existing oscillation and accumulation workflow transfers to additional datasets and physical domains. The released `v0.1.0b1` tag is frozen.

Work on `beta/v0.1.x` may:

- add stable datasets and cross-domain demonstrations;
- evaluate oscillation and accumulation behavior under new conditions;
- compare object schemas, measurements, robustness, and failure modes;
- strengthen tests, provenance, and reproducibility;
- correct defects without silently changing released semantics;
- extend the beta research record with evidence from this line of research.

Architectural redesign, successor object models, and incompatible API development belong on the `featuregraph/featuregraph` `main` branch. The distinction is between extending the beta's empirical reach and extending its architecture.

## Citation

If you use FeatureGraph in research, cite the archived beta software record:

> Habib, N. (2026). *FeatureGraph* (v0.1.0b1). Zenodo. https://doi.org/10.5281/zenodo.21984186

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## License

FeatureGraph is released under the MIT License.
