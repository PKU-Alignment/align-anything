Overview
========

We support basic alignment algorithms, *i.e.,*, supervised finetuning (SFT), direct preference optimization (DPO) and proximal policy optimization (PPO). Our implementation covered different modalities, each of which may involve additional algorithms. For instance, in the ``Text -> Text`` modality, we have also implemented SimPO, KTO, and others.

==================================== === == === ===
Modality                             SFT RM DPO PPO
==================================== === == === ===
``Text -> Text (t2t)``               ✔️  ✔️ ✔️  ✔️
``Text+Image -> Text (ti2t)``        ✔️  ✔️ ✔️  ✔️
``Text+Image -> Text+Image (ti2ti)`` ✔️  ✔️ ✔️  ✔️
``Text -> Image (t2i)``              ✔️  ⚒️ ✔️  ⚒️
``Text -> Video (t2v)``              ✔️  ⚒️ ✔️  ⚒️
``Text -> Audio (t2a)``              ✔️  ⚒️ ✔️  ⚒️
==================================== === == === ===

.. note::

    Align-Anything employs a highly scalable implementation style through inheritance and derivation. Researchers can quickly extend algorithms such as SimPO and KTO to other modalities. We hope this will bring convenience to researchers.
