Template and Dataset Custumization
==================================

We offer a highly scalable dataset registration interface,
enabling users to embed customized datasets simply by designing and
specifying their `dataset_formatter.py <https://github.com/PKU-Alignment/align-anything/blob/main/align_anything/configs/format_dataset.py>`__.

Taking `Align-Anything-200K (Text+Image -> Text, short as AA_TI2T) <https://huggingface.co/datasets/PKU-Alignment/align-anything>`__ as an
example, we illustrate here how to design the template and incorporate
it into a complete RLHF workflow.

The orignal data key-value pairs for AA_TI2T are as follows:

.. code:: python

   {
     'image': PIL.Image.Image,
     'question': str,
     'response_1': str,
     'response_2': str,
     'overall_response': int,
   }

We first need to create a new template named ``AA_TI2T`` for this dataset
(we use ``_`` here because it is more pythonic), and specify the
required parameters such as system_prompt.

.. code:: python

    @register_template('AA_TI2T')
    class AA_TI2T(BaseFormatter):
        system_prompt: str = ""

Then, we can implement following three types of functions to finish the dataset registration:

+-----------------------------------+-----------------------------------+
| Type                              | Description                       |
+===================================+===================================+
| ``format_supervised_sample``      | Mapping the dataset to the        |
|                                   | supervised training format (For   |
|                                   | SFT).                             |
+-----------------------------------+-----------------------------------+
| ``format_preference_sample``      | Mapping the dataset to the        |
|                                   | preference training format (For   |
|                                   | RM, DPO, KTO, *etc.*).            |
+-----------------------------------+-----------------------------------+
| ``format_prompt_only_sample``     | Mapping the dataset to the unique |
|                                   | prompt only training format (For  |
|                                   | PPO).                             |
+-----------------------------------+-----------------------------------+

General Format
~~~~~~~~~~~~~~

Our ``dataset_formatter`` is designed to be a ``conversation`` format, because it can be naturally supported by the ``apply_chat_template`` function in ``transformers`` (more details can be found in `this tutorial <https://huggingface.co/docs/transformers/main/chat_templating>`__).

.. hint::

    An example of conversation format is as follows:

    .. code:: python

       [
           {'role': 'user', 'content': [
                   {'type': 'image'},
                   {'type': 'text', 'text': prompt},
               ]
           },
           {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
       ]

Another important element is the ``multi_modal_info`` field, which is used to store the multi-modal information of the dataset, an example is as follows:

.. code:: python

   {
     'image': PIL.Image.Image,
   }

Format Supervised Sample
~~~~~~~~~~~~~~~~~~~~~~~~

The ``format_supervised_sample`` function is used to convert the dataset to the Q-A format, an example is as follows:

.. code:: python

    @register_template('AA_TI2T')
    class AA_TI2T(BaseFormatter):
        system_prompt: str = ""

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['prompt']
        answer = raw_sample['response']
        image = raw_sample['image'].convert('RGBA')

        return [
            {'role': 'user', 'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ]
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
        ], {'image': image}


Format Preference Sample
~~~~~~~~~~~~~~~~~~~~~~~~

The ``format_preference_sample`` function is used to convert the dataset to the preference training format, an example is as follows:

.. code:: python

    @register_template('AA_TI2T')
    class AA_TI2T(BaseFormatter):
        system_prompt: str = ""

        def format_preference_sample(self, raw_sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
            better_id = int(raw_sample['overall_response'])
            worse_id = 2 if better_id==1 else 1

            if better_id not in [1, 2] or worse_id not in [1, 2]:
                return [], [], {}

            raw_better_response = raw_sample[f'response_{better_id}']
            raw_worse_response = raw_sample[f'response_{worse_id}']
            prompt = raw_sample['question']
            image = raw_sample['image'].convert('RGBA')
            better_conversation = [
                {'role': 'user', 'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': prompt},
                    ]
                },
                {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_better_response}]},
            ]
            worse_conversation = [
                {'role': 'user', 'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': prompt},
                    ]
                },
                {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_worse_response}]},
            ]

            meta_info = {
                'image': image,
                'better_response': raw_better_response,
                'worse_response': raw_worse_response,
            }

            return better_conversation, worse_conversation, meta_info


.. note::

    The ``format_preference_sample`` function determines which response is better based on the ``chosen`` or ``rejected``, or other preference labels. Then it will return them as dictionaries with key: ``better_response`` and ``worse_response``.

Format Prompt Only Sample
~~~~~~~~~~~~~~~~~~~~~~~~~

During the RL fine-tuning phase, the model requires generation based on
prompts within the dataset. So the ``format_prompt_only_sample`` function is used to convert the dataset to the prompt only training format, an example is as follows:

.. code:: python

    @register_template('AA_TI2T')
    class AA_TI2T(BaseFormatter):
        system_prompt: str = ""

        def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            prompt = raw_sample['question']
            image = raw_sample['image'].convert('RGBA')

            return [
                {'role': 'user', 'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': prompt},
                    ]
                },
            ], {'image': image}

Conclusion
~~~~~~~~~~

For each modality we have implemented at least one ``dataset_formatter`` as examples at `dataset_formatter.py <https://github.com/PKU-Alignment/align-anything/blob/main/align_anything/configs/format_dataset.py>`__. You can refer to these examples to implement your own dataset formatter.