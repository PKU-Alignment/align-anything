Template and Dataset Custumization
==================================

Align-Anything offers a highly scalable dataset registration interface,
enabling users to embed customized datasets simply by designing and
specifying their `template.py <https://github.com/PKU-Alignment/align-anything/blob/main/align_anything/configs/template.py>`__.

Taking `SPA-VL <https://huggingface.co/datasets/sqrti/SPA-VL>`__ as an
example, we illustrate here how to design the template and incorporate
it into a complete RLHF workflow.

The orignal data key-value pairs for SPA-VL are as follows:

.. code:: python

   {
     'image': '...',
     'question': '...',
     'chosen': '...',
     'rejected': '...',
   }

We first need to create a new template named ``SPA_VL`` for this dataset
(we use ``_`` here because it is more pythonic), and specify the
required parameters such as system_prompt.

.. code:: python

   @register_template('SPA_VL')
   class SPA_VL:
       system_prompt: str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
       user_prompt: str = 'USER: \n<image> {input}'
       assistant_prompt: str = '\nASSISTANT: {output}'
       split_token: str = 'ASSISTANT:'

       def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
           better_response = raw_sample['chosen']
           worse_response = raw_sample['rejected']
           prompt = raw_sample['question']
           image = raw_sample['image']

           formatted_prompt = (
               f'{self.system_prompt}'
               f'{self.user_prompt.format(input=prompt)}'
           )
           formatted_better_output = (
               f'{self.assistant_prompt.format(output=better_response)}'
           )
           formatted_worse_output = (
               f'{self.assistant_prompt.format(output=worse_response)}'
           )
           image = image.convert('RGBA')

           return {
               'prompt': formatted_prompt,
               'better_text': formatted_better_output,
               'worse_text': formatted_worse_output,
               'image': image,
           }

       def check_equal(self, raw_sample: dict[str, Any]) -> bool:
           return raw_sample['chosen'] == raw_sample['rejected']

       def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
           prompt = raw_sample['question'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
           image = raw_sample['image']

           formatted_prompt = (
               f'{self.system_prompt}'
               f'{self.user_prompt.format(input=prompt)}'
               f'{self.assistant_prompt.format(output="")}'
           )
           image = image.convert('RGBA')

           return {
               'text': formatted_prompt,
               'image': image,
           }

Reward modeling
~~~~~~~~~~~~~~~

The reward modeling requires the user to provide a dictionary with data
keys as follows:

.. code:: python

   {
     'prompt': '...',
     'image': '...',
     'better_text': '...',
     'worse_text': '...',
   }

Therefore, the user needs to implement a key-value transformation logic
in ``align-anything/configs/template.py``, for instance, in this case:

.. code:: python

   @register_template('SPA_VL')
   class SPA_VL:
       system_prompt: str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
       user_prompt: str = 'USER: \n<image> {input}'
       assistant_prompt: str = '\nASSISTANT: {output}'
       split_token: str = 'ASSISTANT:'

       def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
           better_response = raw_sample['chosen']
           worse_response = raw_sample['rejected']
           prompt = raw_sample['question']
           image = raw_sample['image']

           formatted_prompt = (
               f'{self.system_prompt}'
               f'{self.user_prompt.format(input=prompt)}'
           )
           formatted_better_output = (
               f'{self.assistant_prompt.format(output=better_response)}'
           )
           formatted_worse_output = (
               f'{self.assistant_prompt.format(output=worse_response)}'
           )
           image = image.convert('RGBA')

           return {
               'prompt': formatted_prompt,
               'better_text': formatted_better_output,
               'worse_text': formatted_worse_output,
               'image': image,
           }

Here, ``format_sample`` parses the keys in the SPA-VL dataset,
determines which response is better based on the ``chosen`` or
``rejected``, and subsequently invokes previously defined parameters
such as ``system_prompt`` to implement the transformation of key-value
pairs.

RL fine-tuning
~~~~~~~~~~~~~~

During the RL fine-tuning phase, the model requires generation based on
prompts within the dataset. Consequently, users need to implement
key-value conversion in ``template.py`` using the following function:

.. code:: python

   @register_template('SPA_VL')
   class SPA_VL:
       system_prompt: str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
       user_prompt: str = 'USER: \n<image> {input}'
       assistant_prompt: str = '\nASSISTANT: {output}'
       split_token: str = 'ASSISTANT:'

       ...  # previous code here

       def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
           prompt = raw_sample['question'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
           image = raw_sample['image']

           formatted_prompt = (
               f'{self.system_prompt}'
               f'{self.user_prompt.format(input=prompt)}'
               f'{self.assistant_prompt.format(output="")}'
           )
           image = image.convert('RGBA')

           return {
               'text': formatted_prompt,
               'image': image,
           }

After designing the aforementioned template, you just need to specify
this template by passing the ``train_template SPA_VL`` argument when
invoking the dataset to complete the corresponding training. Perhaps the
above example still lacks specificity; therefore, we provide command
references that encompass various models executing multiple algorithms
on diverse datasets.

.. note::

    You can expedite your training process by directly running or modifying these scripts `here <./examples/>`__. For special task including ``Text Image Interleaved Input and Output`` and ``Any -> Text``, you can refer to `projects <./projects/>`__.