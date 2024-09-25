Chameleon Plus Fine-Tuning Pipeline
===================================

[`🏠 Homepage <https://github.com/PKU-Alignment/align-anything>`__] [`🤗
AA-Chameleon-7B-Base
Model <https://huggingface.co/PKU-Alignment/AA-chameleon-7b-base>`__]
[`🤗 AA-Chameleon-7B-Plus
Model <https://huggingface.co/PKU-Alignment/AA-chameleon-7b-plus>`__]

Highlights
----------

-  We train the original
   `Chameleon-7B <https://huggingface.co/facebook/chameleon-7b>`__ model
   using a 4.6k subset from
   `laion-art <https://huggingface.co/datasets/fantasyfish/laion-art>`__,
   and acquire the
   `AA-Chameleon-7B-Base <https://huggingface.co/PKU-Alignment/AA-chameleon-7b-base>`__
   model (AA refers to Align-Anything) with image generation
   capabilities.
-  We then use our text-image interleaved input & output dataset and
   `Align-Anything <https://github.com/PKU-Alignment/align-anything>`__
   framework and algorithm to train this model, acquiring a
   `AA-Chameleon-7B-Plus <https://huggingface.co/PKU-Alignment/AA-chameleon-7b-plus>`__
   model, which is much better in text-image interleaved i/o task.

Pipeline
--------

Environment setup
~~~~~~~~~~~~~~~~~

Currently, the official Transformer repo does not support Chameleon
model with image output (see `this
PR <https://github.com/huggingface/transformers/pull/32013>`__ for more
details), so we rely on a certain fork of the repo.

After installing Align-Anything and correctly set up the envrionment,
you can install the forked stable version of the repo by running:

.. code:: bash

   pip install git+https://github.com/htlou/transformers.git@hantao_stable_cham

Pre-Tokenization
~~~~~~~~~~~~~~~~

As chameleon expands the <image> in the input into image placeholders
(<image> \* 1024) in the processor, but add the real image tokens inside the forward function, we need to pretokenize the input and merge the real image tokens to the labels before doing the training.

.. tab-set::

    .. tab-item:: Supervised, Sequential

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Pretokenization for SFT
            ^^^^^^^^^^^^^^^^^^^^^^^^
            Set the template and the input key name correctly in `pre_tokenize_example.py <https://github.com/PKU-Alignment/align-anything/blob/main/projects/text_image_to_text_image/pre_tokenize_example.py>`__

            and run:

            .. code-block:: bash

                python pre_tokenize_example.py --model_path $MODEL_PATH --input_path $INPUT_PATH --output_path $OUTPUT_PATH

            Replace ``$MODEL_PATH``, ``$INPUT_PATH`` and ``$OUTPUT_PATH`` with the correct paths.
    
    .. tab-item:: Supervised, Parallel

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Pretokenization for SFT, Accelerated
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            If you have multiple GPUs, you can use the accelerated version of the scriptto speed up the process.

            Set the template, the input key name, and the number of processes and GPUs correctly in `pre_tokenize_parallel_example.py <https://github.com/PKU-Alignment/align-anything/blob/main/projects/text_image_to_text_image/pre_tokenize_parallel_example.py>`__

            and run:

            .. code-block:: bash

                python pre_tokenize_parallel_example.py --model_path $MODEL_PATH --input_path $INPUT_PATH --output_path $OUTPUT_PATH --cache_dir $CACHE_DIR

            Replace ``$MODEL_PATH``, ``$INPUT_PATH``, ``$OUTPUT_PATH`` and ``$CACHE_DIR`` with the correct paths.

            .. admonition:: Note
                :class: note

                - Despite utilizing GPUs, the primary bottleneck in pre-tokenization is the CPU's capacity. It is crucial to adjust the number of processes and GPUs based on your CPU's capabilities to ensure efficient performance.

                - The parallel pre-tokenization process incorporates a caching mechanism to reduce memory load. You should specify a directory for the cache using the ``--cache_dir`` option, where the cache data will be stored.

    .. tab-item:: Preference

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Pretokenization for DPO or RM
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            If you are dealing with prefernce dataset (for DPO or RM), set the template, the input key name, and the number of processes and GPUs correctly in
            `preference_tokenize_example.py <https://github.com/PKU-Alignment/align-anything/blob/main/projects/text_image_to_text_image/preference_tokenize_example.py>`__
            and run:

            .. code:: bash

                python preference_tokenize_example.py --model_path $MODEL_PATH --input_path $INPUT_PATH --output_path $OUTPUT_PATH --cache_dir $CACHE_DIR

            Replace ``$MODEL_PATH``, ``$INPUT_PATH``, ``$OUTPUT_PATH`` and ``$CACHE_DIR`` with the correct paths.

            .. admonition:: Note
                :class: note

                - By default, we use parallel pre-tokenization to speed up the process here. If you want to fall back to sequential pre-tokenization, you can set the `num_processes` and `num_gpus` to 1.

    .. tab-item:: Prompt Only

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Pretokenization for PPO
            ^^^^^^^^^^^^^^^^^^^^^^^

            If you are dealing with prompt only dataset (for PPO), set the template, the input key name, and the number of processes and GPUs correctly in
            `prompt_only_tokenize_example.py <https://github.com/PKU-Alignment/align-anything/blob/main/projects/text_image_to_text_image/prompt_only_tokenize_example.py>`__
            and run:

            .. code:: bash

                python prompt_only_tokenize_example.py --model_path $MODEL_PATH --input_path $INPUT_PATH --output_path $OUTPUT_PATH --cache_dir $CACHE_DIR

            Replace ``$MODEL_PATH``, ``$INPUT_PATH``, ``$OUTPUT_PATH`` and ``$CACHE_DIR`` with the correct paths.

            .. admonition:: Note
                :class: note

                - By default, we use parallel pre-tokenization to speed up the process here. If you want to fall back to sequential pre-tokenization, you can set the `num_processes` and `num_gpus` to 1.

Training Model
~~~~~~~~~~~~~~

After pre-tokenizing the dataset, you can start training the model using the following scripts.

.. tab-set::

    .. tab-item:: SFT

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Supervised Fine-Tuning
            ^^^^^^^^^^^^^^^^^^^^^^

            Add a script named ``sft_text_image_to_text_image.sh`` under the ``scripts`` file like this:

            .. code:: bash

                MODEL_NAME_OR_PATH="PKU-Alignment/AA-chameleon-7b-base"
                TRAIN_DATASETS=""
                PT_NAME=""
                OUTPUT_DIR="../outputs/sft_text_image_to_text_image"
                export WANDB_API_KEY=""
                source ./setup.sh

                deepspeed \
                    --master_port ${MASTER_PORT} \
                    --module align_anything.trainers.text_image_to_text_image.sft \
                    --model_name_or_path ${MODEL_NAME_OR_PATH} \
                    --train_datasets ${TRAIN_DATASETS} \
                    --train_data_files ${PT_NAME} \ 
                    --output_dir ${OUTPUT_DIR} \
                    --train_template ANYTHING_TI2TI \
                    --train_split 'train' \
                    --per_device_train_batch_size 2 \
                    --per_device_eval_batch_size 2 \
                    --gradient_accumulation_steps 2 \
                    --save_interval 500 \
                    --learning_rate 5e-5 \
                    --epochs 3 \
                    --lr_scheduler_type constant
            
            and set up the correct model path and dataset path, then run:

            .. code:: bash

                bash scripts/sft_text_image_to_text_image.sh

            .. admonition:: Note
                :class: note

                - Supposed your pre-tokenized dataset is stored in ``/path/to/dataset/dataset_file_name.pt``, then the ``TRAIN_DATASETS`` should be ``/path/to/dataset`` and the ``PT_NAME`` should be ``dataset_file_name.pt``.

    .. tab-item:: DPO

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Direct Preference Optimization
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            Add a script named ``dpo_text_image_to_text_image.sh`` under the ``scripts`` file like this:

            .. code:: bash

                MODEL_NAME_OR_PATH="PKU-Alignment/AA-chameleon-7b-base"
                TRAIN_DATASETS=""
                PT_NAME=""
                OUTPUT_DIR="../outputs/dpo_text_image_to_text_image"
                export WANDB_API_KEY=""
                source ./setup.sh

                deepspeed \
                    --master_port ${MASTER_PORT} \
                    --module align_anything.trainers.text_image_to_text_image.dpo \
                    --model_name_or_path ${MODEL_NAME_OR_PATH} \
                    --train_datasets ${TRAIN_DATASETS} \
                    --train_data_files ${PT_NAME} \
                    --output_dir ${OUTPUT_DIR} \
                    --train_template ANYTHING_TI2TI \
                    --train_split 'train' \
                    --per_device_train_batch_size 2 \
                    --per_device_eval_batch_size 2 \
                    --gradient_accumulation_steps 2 \
                    --save_interval 2500 \
                    --learning_rate 5e-7 \
                    --epochs 3 \
                    --lr_scheduler_type cosine 

            and set up the correct model path and dataset path, then run:

            .. code:: bash

                bash scripts/sft_text_image_to_text_image.sh

            .. admonition:: Note
                :class: note

                - Supposed your pre-tokenized dataset is stored in ``/path/to/dataset/dataset_file_name.pt``, then the ``TRAIN_DATASETS`` should be ``/path/to/dataset`` and the ``PT_NAME`` should be ``dataset_file_name.pt``.

    .. tab-item:: RM

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Reward Model
            ^^^^^^^^^^^^

            Add a script named ``rm_text_image_to_text_image.sh`` under the ``scripts`` file like this:

            .. code:: bash

                MODEL_NAME_OR_PATH="PKU-Alignment/AA-chameleon-7b-base"
                TRAIN_DATASETS=""
                TRAIN_PT_NAME="" 
                EVAL_DATASETS=""
                EVAL_PT_NAME="" 
                OUTPUT_DIR="../outputs/rm_text_image_to_text_image"
                export WANDB_API_KEY=""
                source ./setup.sh

                deepspeed \
                    --master_port ${MASTER_PORT} \
                    --module align_anything.trainers.text_image_to_text_image.rm \
                    --model_name_or_path ${MODEL_NAME_OR_PATH} \
                    --train_datasets ${TRAIN_DATASETS} \
                    --output_dir ${OUTPUT_DIR} \
                    --per_device_train_batch_size 2 \
                    --per_device_eval_batch_size 2 \
                    --gradient_accumulation_steps 2 \
                    --train_template ANYTHING_TI2TI \
                    --train_split train \
                    --train_data_files ${TRAIN_PT_NAME} \
                    --eval_datasets ${EVAL_DATASETS} \
                    --eval_data_files ${EVAL_PT_NAME} \
                    --eval_template ANYTHING_TI2TI \
                    --learning_rate 5e-6 \
                    --epochs 3 \
                    --lr_scheduler_type cosine \
                    --save_interval 2500 
                
            and set up the correct model path and dataset path, then run:

            .. code:: bash

                bash scripts/rm_text_image_to_text_image.sh

            .. admonition:: Note
                :class: note

                - Supposed your pre-tokenized dataset is stored in ``/path/to/dataset/dataset_file_name.pt``, then the ``TRAIN_DATASETS`` should be ``/path/to/dataset`` and the ``PT_NAME`` should be ``dataset_file_name.pt``. Same for ``EVAL_DATASETS`` and ``EVAL_PT_NAME``.

    .. tab-item:: PPO

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Proximal Policy Optimization
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            Add a script named ``ppo_text_image_to_text_image.sh`` under the ``scripts`` file like this:

            .. code:: bash

                ACTOR_MODEL_NAME_OR_PATH="PKU-Alignment/AA-chameleon-7b-base"
                CRITIC_MODEL_NAME_OR_PATH=""
                REWARD_MODEL_NAME_OR_PATH=""
                TRAIN_DATASETS=""
                TRAIN_PT_NAME="" 
                PTX_DATASETS=""
                PTX_PT_NAME="" 
                OUTPUT_DIR="../outputs/ppo_text_image_to_text_image"

                source ./setup.sh

                deepspeed \
                --master_port ${MASTER_PORT} \
                --module align_anything.trainers.text_image_to_text_image.ppo \
                --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
                --reward_model_name_or_path ${REWARD_MODEL_NAME_OR_PATH} \
                --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
                --train_datasets ${TRAIN_DATASETS} \
                --train_template ANYTHING_TI2TI \
                --train_data_files ${TRAIN_PT_NAME} \
                --ptx_datasets ${PTX_DATASETS} \
                --ptx_data_files ${PTX_PT_NAME} \
                --ptx_template Llava \
                --output_dir ${OUTPUT_DIR}

            and set up the correct model path and dataset path, then run:

            .. code:: bash

                bash scripts/ppo_text_image_to_text_image.sh

            .. admonition:: Note
                :class: note

                - The ``CRITIC_MODEL_NAME_OR_PATH`` and ``REWARD_MODEL_NAME_OR_PATH`` should be the path to your reward model.

                - Supposed your pre-tokenized dataset is stored in ``/path/to/dataset/dataset_file_name.pt``, then the ``TRAIN_DATASETS`` should be ``/path/to/dataset`` and the ``TRAIN_PT_NAME`` should be ``dataset_file_name.pt``. Same for ``PTX_DATASETS`` and ``PTX_PT_NAME``.

Model Evaluation
----------------

Batch Inference
~~~~~~~~~~~~~~~

Currently the batch inference of Chameleon is not integrated into the
Align-Anything repo, so we need to use `another repo <https://github.com/htlou/mmsg_chameleon>`__. Here’s a
forked (and revised to make it stable) version:

.. code:: bash

   git clone https://github.com/htlou/mmsg_chameleon.git
   cd mmsg_chameleon

Then set up the envrionment using

.. code:: bash

   pip install -e . 

After setting up the envrioment, set up the correct paths in
``scripts/interleaved_gen.sh`` and then run

.. code:: bash

   bash scripts/interleaved_gen.sh

to do batch inference.

GPT-based Evaluation
~~~~~~~~~~~~~~~~~~~~

Currently the GPT-based evaluation of text-image interleaved messages is
not integrated into the Align-Anything repo, so we need to use `another
repo <https://github.com/htlou/gpt4_eval>`__:

.. code:: bash

   git clone https://github.com/htlou/gpt4_eval.git
   cd gpt4_eval

You can set the ``INPUT_TYPE`` in the ``script.sh`` to
``interleaved-compare`` and run:

.. code:: bash

   bash script.sh

to do the evaluation.

Model Performance
-----------------

We trained the lm_head of Chameleon-7B model using the full data from
`laisonart <huggingface.co/datasets/fantasyfish/laion-art>`__, and
acquired a AA-Chameleon-7B-base model (AA refers to Align-Anything) with image generation capabilities. Then, we used our text-image interleaved input & output dataset to finetune this model, acquiring a AA-Chameleon-7B-plus, which is much better in text-image interleaved i/o task. The model is trained for 3 epochs with a batch size of 2 and a learning rate of 5e-05 (similar to meta’s setting in `their paper <https://ai.meta.com/blog/generative-ai-text-images-cm3leon>`__).

GPT-4o evaluation
~~~~~~~~~~~~~~~~~

We compared the performance of AA-Chameleon-7B+ with Anole-7b model on a
text-image interleaved i/o dataset (with a length of 500). Here’s the
results:

=== ==== ===== =============================
win lose equal win rate (not counting equal)
=== ==== ===== =============================
240 175  85    57.8%
=== ==== ===== =============================

Generation Examples
~~~~~~~~~~~~~~~~~~~

.. tab-set::

    .. tab-item:: #1

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Example #1
            ^^^^^^^^^

            Input:

            .. toggle:: Click here for more information about the input

                **Input text:**  

                Add a vintage filter to this bathroom image to give it a retro look and describe the steps involved in applying the filter.

                **Input image:**  

                |image|


            Output (before alignment):

            .. toggle:: Click to see the output before alignment

               **Output text (before alignment):**  

               Vintage look. Filters like this are designed to give old, vintage items a new life by giving them a distressed, nostalgic look. This process involves removing small particles from the material to create a rough, textured effect.

               **Output image (before alignment):**  

               None


            Output (after alignment):

            .. toggle:: Click to see the output after alignment

               **Output text (after alignment):**  

               I have applied a vintage filter to the bathroom image to give it a retro look. The process involved adjusting the color temperature to add a slight cool tone, reducing the saturation to give it a more muted appearance, and increasing the contrast to enhance the vintage feel.

               **Output image (after alignment):**  

               |image1|

    .. tab-item:: #2

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Example #2
            ^^^^^^^^^

            Input:

            .. toggle:: Click here for more information about the input

                **Input text:**  

                Create a detailed recipe for making the pastry shown in the image, including ingredients, preparation steps, and baking instructions.

                **Input image:**  

                |image2|
            
            Output (before alignment):

            .. toggle:: Click to see the output before alignment

               **Output text (before alignment):**  

               Absolutely! Here is a detailed recipe for making the pastry in the image:\\n\\nIngredients:\\n\\n* 1 cup all-purpose flour\\n* 1/2 cup cold unsalted butter, cut into small pieces\\n* 1/4 cup ice-cold water\\n* 1 egg, lightly beaten\\n* 1 tablespoon honey\\n* 1/4 teaspoon salt\\n

               **Output image (before alignment):**  

               None

            Output (after alignment):

            .. toggle:: Click to see the output after alignment

               **Output text (after alignment):**  

               <image>\\n\\nCooking Instructions:\\n1. Preheat Oven: Preheat your oven to 350\\u00b0F (175\\u00b0C).\\n2. Prepare Filling: In a small bowl, mix well the flour, sugar, eggs, and any other desired ingredients. To make it easier to handle, you can place the bowl of the mixture on a baking tray lined with parchment paper.\\n3. Bake Donut: Place the prepared donut on the preheated baking tray. Bake for 8-10 minutes until the donut is golden brown and crispy on top.\\n4. Cool and Serve: Allow the donut to cool slightly before serving. You can garnish it with a drizzle of chocolate sauce or caramel sauce if desired.\\n\\nEnjoy your homemade chocolate-filled donut!

               **Output image (after alignment):**  
               |image3|

    .. tab-item:: #3

        .. card::
            :class-header: sd-bg-success sd-text-white sd-font-weight-bold
            :class-card: sd-outline-success sd-rounded-1 sd-font-weight-bold
            :class-footer: sd-font-weight-bold

            Example #3
            ^^^^^^^^^

            Input:

            .. toggle:: Click here for more information about the input

                **Input text:**  

                Enhance the presentation of these pizzas with gourmet toppings and plating, and describe the steps involved.

                **Input image:**  
                |image4|

            Output (before alignment):

            .. toggle:: Click to see the output before alignment

               **Output text (before alignment):**  

               The steps involved in presenting these homemade pizzas with gourmet toppings and plating them are as follows:\n\n1. **Gather Ingredients:**\n   - Use high-quality, fresh ingredients: chees, vegetables, fruits, and meats.\n   - Avoid using pre-concealed toppings: Instead, let your creations shine.\n\n2. **Mix Ingredients:**\n   - Mix all ingredients thoroughly but avoid mishandling them.\n   - Cover the mixing bowl with cling wrap to prevent cross-contamination.\n\n3. **Set Up the Presentation:**\n   - Transfer the mixture to a clean, oven-proof dish and preheat the oven to 400\u00b0F (200\u00b0C).\n   - Cut the pizza into appropriate portions and allow them to cool a few minutes.\n   - Plate the pizzas on a clean, clear surface to showcase their presentation.\n\n4. **Garnish and Presentation:**\n   - Add desired garnishes and toppings to add visual appeal and flavor.\n   - Secure the plating board with a clear, decorative overlay to protect the food from elements like moisture.\n\n5. **Serve and Enjoy:**\n   - Ensure the pizzas are hot and fresh to avoid any bacterial risks.\n   - Smooth any uneven edges and serve with a complementary dipping sauce or condiments.\n\nI've described the steps involved in preparing these homemade pizzas and their plating process.<image>

               **Output image (before alignment):**  
               |image6|

            Output (after alignment):

            .. toggle:: Click to see the output after alignment

                **Output text (after alignment):**  
                
                I have enhanced the presentation of the pizzas by adding gourmet toppings. The pizzas now feature fresh basil leaves, prosciutto, cherry tomatoes, shaved Parmesan cheese, and a drizzle of balsamic glaze. The toppings are arranged artistically on each pizza to create a visually appealing and gourmet presentation.

                
                **Output image (after alignment):**  

                |image5|


.. |image| image:: https://github.com/PKU-Alignment/align-anything/blob/main/assets/text_image_to_text_image/bathroom.jpg?raw=true
.. |image1| image:: https://github.com/PKU-Alignment/align-anything/blob/main/assets/text_image_to_text_image/bathroom_generated.png?raw=true
.. |image2| image:: https://github.com/PKU-Alignment/align-anything/blob/main/assets/text_image_to_text_image/baking.jpg?raw=true
.. |image3| image:: https://github.com/PKU-Alignment/align-anything/blob/main/assets/text_image_to_text_image/baking_generated.png?raw=true
.. |image4| image:: https://github.com/PKU-Alignment/align-anything/blob/main/assets/text_image_to_text_image/pizza.jpg?raw=true
.. |image5| image:: https://github.com/PKU-Alignment/align-anything/blob/main/assets/text_image_to_text_image/pizza_generated.png?raw=true
.. |image6| image:: https://github.com/htlou/image_bed/blob/main/pizza_before.png?raw=true