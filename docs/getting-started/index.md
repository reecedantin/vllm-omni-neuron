# Get started

Set up vLLM Omni Neuron with manual installation or a Neuron DLC, then generate your first video on AWS Trainium.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Setup guide
:link: setup-guide
:link-type: doc

Choose manual or DLC installation, configure persistent caches, and verify the Neuron environment.
:::

:::{grid-item-card} Online serving quickstart
:link: quickstart-online-serving-wan22
:link-type: doc

Launch an OpenAI-compatible video server and send your first request.
:::

:::{grid-item-card} Offline serving quickstart
:link: quickstart-offline-serving-wan22
:link-type: doc

Run batch video generation with the offline Omni engine from Python.
:::

:::{grid-item-card} Cosmos3-Edge offline quickstart (Inferentia2)
:link: quickstart-offline-serving-cosmos3-edge
:link-type: doc

Generate an image, a video, or a robot action chunk with Cosmos3-Edge on inf2.
:::

::::

:::{toctree}
:maxdepth: 1
:hidden:

Setup guide <setup-guide>
Online serving quickstart <quickstart-online-serving-wan22>
Offline serving quickstart <quickstart-offline-serving-wan22>
Cosmos3-Edge offline quickstart <quickstart-offline-serving-cosmos3-edge>
:::
