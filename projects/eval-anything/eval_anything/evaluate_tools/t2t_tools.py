# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Evaluation tools package (including calculation of various metrics and extraction of specified patterns)
"""

import re

# from latex2sympy2 import latex2sympy
from math import *
from typing import Iterable, List, Union

from eval_anything.evaluate_tools.base_tools import BaseTool
from eval_anything.utils.register import AnswerExtractorRegistry, JudgeRegistry


###### Answer Extractor ######


@AnswerExtractorRegistry.register('regex_match')
class RegexMatch(BaseTool):
    def __init__(self, pattern: str, match_index: int = None):
        self.pattern = pattern
        self.match_index = match_index

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        """
        Match the specified pattern in the text.
        Args:
            data (list/iterable): the text to be matched
        Returns:
            list/None: the matched result
        """

        def match_text(text):
            import re

            pattern = re.compile(self.pattern)
            match = list(pattern.finditer(text))
            if match:
                if self.match_index is not None:
                    return match[self.match_index].group()
                else:
                    return match.group()
            else:
                return None

        matches = [match_text(item) for item in data]
        return matches

    def __call__(self, data: Union[List[str], Iterable[str]]) -> List[Union[str, None]]:
        return self.apply(data)


@AnswerExtractorRegistry.register('regex_match_number')
class RegexMatchNumber(RegexMatch):
    def __init__(self, additional_pattern: str = None, match_index: int = None):
        pattern_match_number = r'(?:[+-]?(?:\d+/\d+|(?:\d*\.\d+)|\d+)|√(?:\([+-]?(?:\d+/\d+|(?:\d*\.\d+)|\d+)\)|[+-]?(?:\d+/\d+|(?:\d*\.\d+)|\d+)))'
        self.pattern = (
            additional_pattern.format(original_pattern=pattern_match_number)
            if additional_pattern
            else pattern_match_number
        )
        self.match_index = match_index

    def apply(self, data: Union[List[str], Iterable[str]]) -> List[Union[str, None]]:
        import re

        from eval_anything.utils.utils import _strip_string, remove_boxed, remove_few_shot_prefix

        def match_text(text):
            if '\\boxed' in text:
                return remove_boxed(text)
            matches = re.findall(self.pattern, text)
            return matches[self.match_index] if matches else None

        return [_strip_string(match_text(remove_few_shot_prefix(item))) for item in data]

    def __call__(self, data: Union[List[str], Iterable[str]]) -> List[Union[str, None]]:
        return self.apply(data)


@AnswerExtractorRegistry.register('regex_match_text')
class RegexMatchText(RegexMatch):
    def __init__(self, additional_pattern: str = None, match_index: int = None):
        # Pattern to match single letter answers A, B, C, D (case insensitive)
        pattern_match_text = r'[A-Da-d]'
        self.pattern = (
            additional_pattern.format(original_pattern=pattern_match_text)
            if additional_pattern
            else pattern_match_text
        )
        self.match_index = match_index

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        """
        Match letter answers in the text and convert to uppercase.
        Args:
            data (list/iterable): the text to be matched
        Returns:
            list/None: the matched result in uppercase
        """
        matches = super().apply(data)
        # Convert matched letters to uppercase for consistency
        return [match.upper() if match else None for match in matches]


@AnswerExtractorRegistry.register('regex_match_letter')
class RegexMatchLetter(RegexMatch):
    def __init__(self, additional_pattern: str = None, match_index: int = None):
        # Base pattern to match letters in parentheses (A-Z)
        pattern_match_letter = r'\(([A-Za-z])\)'  # Capture the letter between parentheses
        self.pattern = (
            additional_pattern.format(original_pattern=pattern_match_letter)
            if additional_pattern
            else pattern_match_letter
        )
        self.match_index = match_index

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        def match_text(text):
            import re

            pattern = re.compile(self.pattern, re.IGNORECASE)
            match = list(pattern.finditer(text))
            if match:
                if self.match_index is not None:
                    # Extract just the letter part (group 1) and convert to uppercase
                    return match[self.match_index].group(1).upper()
                else:
                    return match.group(1).upper()
            else:
                return None

        matches = [match_text(item) for item in data]
        return matches


@AnswerExtractorRegistry.register('regex_match_code')
class RegexMatchCode(RegexMatch):
    def __init__(
        self, additional_pattern: str = None, match_index: int = None, language: str = 'python'
    ):
        # Pattern to match code blocks between ```python ``` or ``` ```
        pattern_match_code = r'```(?:{language})?\s*([\s\S]*?)\s*```'
        self.pattern = (
            additional_pattern.format(original_pattern=pattern_match_code)
            if additional_pattern
            else pattern_match_code
        )
        self.match_index = match_index
        self.language = language

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        def match_text(text):
            import re

            # Find all positions of ```language and ``` markers
            language_pattern = fr'```{self.language}'
            close_pattern = r'```'
            language_positions = [m.start() for m in re.finditer(language_pattern, text)]
            close_positions = [m.start() for m in re.finditer(close_pattern, text)]

            if not language_positions or not close_positions:
                return ''

            # Match each ```language with its next ``` marker
            code_blocks = []
            for lang_start in language_positions:
                # Find the next closing marker after this language marker
                next_close = None
                for close_pos in close_positions:
                    if close_pos > lang_start:
                        next_close = close_pos
                        break

                if next_close is not None:
                    block_content = text[lang_start : next_close + 3]  # Include the closing ```
                    # Clean up the content
                    content = re.sub(fr'^```{self.language}\s*', '', block_content)
                    content = re.sub(r'\s*```$', '', content)
                    code_blocks.append(content)

            if not code_blocks:
                return ''

            # Return the specific match based on index
            if self.match_index is not None:
                if self.match_index < 0:
                    self.match_index = len(code_blocks) + self.match_index
                if 0 <= self.match_index < len(code_blocks):
                    return code_blocks[self.match_index]
                return ''
            else:
                return code_blocks[0]  # Default to first match

        matches = [match_text(item) for item in data]
        return matches


# refer: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/main_parse_and_eval.py
@AnswerExtractorRegistry.register('regex_match_multi_open')
class RegexMatchMultiOpen(RegexMatch):
    """
    Handle multi-choice and open-ended mixed tasks.
    """

    def __init__(
        self, additional_pattern: str = None, match_index: int = None, letter_pattern: str = None
    ):
        super().__init__(additional_pattern, match_index)
        self.letter_pattern = (
            letter_pattern if letter_pattern else r'\b([A-D])\b|\((?P<letter>[A-D])\)'
        )
        self.indicators_of_keys = [
            'could be ',
            'so ',
            'is ',
            'thus ',
            'therefore ',
            'final ',
            'answer ',
            'result ',
        ]

    def _match_letter(self, text: str):
        """Match the last uppercase letter in the string, either in parentheses or standalone.

        Args:
            text (str): Input text
            match_index (int): Index of the matched letter
        Returns:
            Optional[str]: Matched letter (uppercase) or None
        """
        try:
            pattern = re.compile(self.letter_pattern, re.IGNORECASE)
            matches = list(pattern.finditer(text))

            if not matches:
                return None

            selected_match = matches[self.match_index]
            if selected_match.group('letter'):
                return selected_match.group('letter').upper()  # From parentheses
            return selected_match.group(1).upper()  # Standalone letter

        except (IndexError, AttributeError, TypeError):
            return None

    def _check_is_number(self, string: str) -> bool:
        """Check if the given string is a number"""
        try:
            float(string.replace(',', ''))
            return True
        except ValueError:
            return False

    def _normalize_str(self, string):
        """
        Normalize the str to lower case and make them float numbers if possible.
        """
        # if number, numerize it.
        string = string.strip()

        is_number = self._check_is_number(string)

        if is_number:
            string = string.replace(',', '')
            string = float(string)
            # leave 2 decimal
            string = round(string, 2)
            return [string]
        else:  # it's likely to be a string
            # lower it
            string = string.lower()
            if len(string) == 1:
                return [' ' + string, string + ' ']  # avoid trivial matches
            return [string]

    def _extract_numbers(self, string):
        """
        Exact all forms of numbers from a string with regex.
        """
        # Pattern for numbers with commas
        pattern_commas = r'-?\b\d{1,3}(?:,\d{3})+\b'
        # Pattern for scientific notation
        pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
        # Pattern for simple numbers without commas
        pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])'

        # Extract numbers with commas
        numbers_with_commas = re.findall(pattern_commas, string)
        # Extract numbers in scientific notation
        numbers_scientific = re.findall(pattern_scientific, string)
        # Extract simple numbers without commas
        numbers_simple = re.findall(pattern_simple, string)

        # Combine all extracted numbers
        all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
        return all_numbers

    def _get_key_subresponses(self, response: str) -> List[str]:
        """Extract key responses from the model output"""
        key_responses = []
        response = response.strip().strip('.').lower()
        sub_responses = re.split(r'\.\s(?=[A-Z])|\n', response)

        for index, resp in enumerate(sub_responses):
            indicators = self.indicators_of_keys + (
                ['='] if index == len(sub_responses) - 1 else []
            )

            shortest_key_response = None
            for indicator in indicators:
                if indicator in resp:
                    current_response = resp.split(indicator)[-1].strip()
                    if not shortest_key_response or len(current_response) < len(
                        shortest_key_response
                    ):
                        shortest_key_response = current_response

            if shortest_key_response and shortest_key_response.strip() not in [
                ':',
                ',',
                '.',
                '!',
                '?',
                ';',
                ':',
                "'",
            ]:
                key_responses.append(shortest_key_response)

        return key_responses if key_responses else [response]

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        """Process responses with both letter matching and open-ended analysis"""
        try:

            def process_single_response(response: str) -> List:
                # Ensure response is string
                if not isinstance(response, str):
                    response = str(response)

                # Match letter
                letter_match = self._match_letter(response)

                # Process as open-ended response
                key_responses = self._get_key_subresponses(response)
                pred_list = key_responses.copy()

                for resp in key_responses:
                    pred_list.extend(self._extract_numbers(resp))

                tmp_pred_list = []
                for pred in pred_list:
                    tmp_pred_list.extend(self._normalize_str(str(pred)))

                final_pred_list = [letter_match]
                final_pred_list.extend(list(set(tmp_pred_list)))

                return final_pred_list

            if not data:
                return []

            return [process_single_response(response) for response in data]
        except Exception as e:
            print(f'Error processing responses: {e}')
            return None

    def __call__(self, data: Union[List, Iterable]) -> Union[List, None]:
        return self.apply(data)


# refer: https://github.com/mathllm/MATH-V/blob/main/evaluation/utils.py
@AnswerExtractorRegistry.register('regex_match_latex_math')
class RegexMatchLatexMath(RegexMatch):
    """
    Handle multi-choice and open-ended mixed tasks in Latex format.
    """

    def __init__(
        self, additional_pattern: str = None, match_index: int = None, letter_pattern: str = None
    ):
        super().__init__(additional_pattern, match_index)

    def apply(self, data: Union[List, Iterable]) -> Union[List, None]:
        """Process responses with both letter matching and open-ended analysis in Latex format."""

        def _fix_fracs(string):
            """Fixes LaTeX fraction formatting by ensuring proper braces for numerators and denominators."""
            substrs = string.split('\\frac')
            new_str = substrs[0]

            if len(substrs) > 1:
                substrs = substrs[1:]

                for substr in substrs:
                    new_str += '\\frac'
                    if len(substr) > 0 and substr[0] == '{':
                        new_str += substr
                    else:
                        try:
                            assert len(substr) >= 2
                        except:
                            return string

                        a = substr[0]  # Numerator
                        b = substr[1]  # Denominator

                        if b != '{':
                            new_str += (
                                '{' + a + '}{' + b + '}' + (substr[2:] if len(substr) > 2 else '')
                            )
                        else:
                            new_str += '{' + a + '}' + b + (substr[2:] if len(substr) > 2 else '')

            return new_str

        def _fix_a_slash_b(string):
            """Convert a/b fraction format to LaTeX \frac{a}{b} format."""

            if len(string.split('/')) != 2:
                return string
            a, b = string.split('/')
            try:
                a = int(a)
                b = int(b)
                assert string == f'{a}/{b}'

                new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
                return new_string
            except:
                return string

        def _remove_right_units(string):
            splits = string.split('\\text{ ')
            return splits[0]

        def _fix_sqrt(string):
            """Fix LaTeX sqrt commands by adding braces around single-character arguments."""
            if '\\sqrt' not in string:
                return string

            splits = string.split('\\sqrt')
            new_string = splits[0]

            for split in splits[1:]:
                if len(split) > 0 and split[0] != '{':
                    a = split[0]
                    new_substr = '\\sqrt{' + a + '}' + split[1:]
                else:
                    new_substr = '\\sqrt' + split
                new_string += new_substr

            return new_string

        def _strip_string(string):
            """Strip and normalize LaTeX strings by removing special characters, normalizing fractions, and standardizing number formats."""
            string = string.replace('\n', '').replace('\\!', '')
            string = string.replace('\\\\', '\\').replace('tfrac', 'frac').replace('dfrac', 'frac')

            replacements = [
                ('\\left', ''),
                ('\\right', ''),
                ('^{\\circ}', ''),
                ('^\\circ', ''),
                ('\\$', ''),
                ('$', ''),
                ('\\%', ''),
                (r'\%', ''),
            ]
            for old, new in replacements:
                string = string.replace(old, new)

            string = _remove_right_units(string)

            string = string.replace(' .', ' 0.').replace('{.', '{0.')

            if len(string) == 0:
                return string
            if string[0] == '.':
                string = '0' + string

            if len(string.split('=')) == 2:
                string = string.split('=')[-1]
            if len(string.split('\\approx')) == 2:
                string = string.split('\\approx')[-1]

            if 'sqrt' in string:
                string = _fix_sqrt(string)

            string = string.replace(' ', '')

            if 'sqrt' in string:
                string = _fix_fracs(string)

            if string == '0.5':
                string = '\\frac{1}{2}'

            string = _fix_a_slash_b(string)

            return string

        def find_math_answer(s: str) -> str:
            """Extract and clean mathematical answer from input string."""
            s = s.lower()
            if '{}' in s:
                s = s.replace('{}', '')

            try:
                pattern = re.compile('oxed{(.*)}', flags=re.S)
                ans = pattern.findall(s)[-1]
            except:
                ans = s

            if ans.find('}') != -1 and (ans.find('{') == -1 or ans.find('}') < ans.find('{')):
                ans = ans.split('}')[0]

            ans = ans.split('=')[-1]
            ans = ans.split('\\approx')[-1]

            ans = ans.replace(' ', '').replace('\\,', '').replace('∞', '\\infty')
            ans = ans.replace(r'+\infty', r'\infty').replace('\\\\', '\\').replace('\n', '')
            ans = ans.replace('\\text', '').replace('\\mbox', '').replace('bmatrix', 'pmatrix')
            ans = ans.replace('\\left', '').replace('\\right', '').replace('^{\\circ}', '')
            ans = ans.replace('^\\circ', '').replace('{m}^3', '').replace('m^3', '')
            ans = (
                ans.replace('{units}', '')
                .replace('units', '')
                .replace('{km}', '')
                .replace('km', '')
            )
            return _strip_string(ans)

        try:

            def process_single_response(response: str) -> List:
                """Process a single response to extract mathematical answer."""
                response = '\\boxed{' + response.split('oxed{')[-1]
                response = (
                    find_math_answer(response)
                    .replace('(a)', 'a')
                    .replace('(b)', 'b')
                    .replace('(c)', 'c')
                    .replace('(d)', 'd')
                    .replace('(e)', 'e')
                    .replace('{a}', 'a')
                    .replace('{b}', 'b')
                    .replace('{c}', 'c')
                    .replace('{d}', 'd')
                    .replace('{e}', 'e')
                    .rstrip('.')
                    .lstrip(':')
                    .strip()
                )
                return response

            return [process_single_response(response) for response in data]
        except Exception as e:
            print(f'Error processing responses: {e}')
            return None

    def __call__(self, data: Union[List, Iterable]) -> Union[List, None]:
        return self.apply(data)


###### Judge Methods ######


@JudgeRegistry.register('judge_equal')
class JudgeEqual(BaseTool):
    def __init__(self):
        super().__init__()

    def apply(self, data_1, data_2) -> bool:
        return data_1 == data_2

    def __call__(self, data_1, data_2) -> bool:
        return self.apply(data_1, data_2)


# refer: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/main_eval_only.py
@JudgeRegistry.register('judge_equal_list')
class JudgeEqualList(BaseTool):
    def __init__(self):
        super().__init__()

    def apply(self, data_1, data_2) -> bool:
        """Compare model answer list with ground truth

        Args:
            data_1: Model answer list
            data_2: Ground truth (str representing float, list of str, or str)

        Returns:
            bool: True if any ground truth matches any answer in model's list
        """
        if data_1 is None:
            return False

        try:
            gold_answer = eval(data_2)
        except:
            gold_answer = data_2

        if isinstance(gold_answer, list):
            correct = False
            for gold_answer_i in gold_answer:
                for pred_answer in data_1:
                    if isinstance(pred_answer, str):
                        if isinstance(gold_answer_i, str):
                            if gold_answer_i in pred_answer:
                                correct = True
                                break
                    else:
                        try:
                            gold_answer_i = float(gold_answer_i)
                        except:
                            gold_answer_i = gold_answer_i
                        if gold_answer_i == pred_answer:
                            correct = True
                            break
            return correct
        else:
            return gold_answer in data_1

    def __call__(self, data_1, data_2) -> bool:
        return self.apply(data_1, data_2)


# refer: https://github.com/mathllm/MATH-V/blob/main/evaluation/utils.py
@JudgeRegistry.register('judge_latex_equal')
class JudgeLatexEqual(BaseTool):
    def __init__(self):
        super().__init__()

    def apply(self, data_1, data_2) -> bool:
        def eval_tuple(s):
            """
            Evaluates the mathematical expressions within tuples or lists represented as strings.

            Args:
                s (str): The string representation of a tuple or list.
                         E.g., "(a,b,c,...)" or "[a,b,c,...]"

            Returns:
                str: A string representation of the tuple or list with evaluated expressions.
                     Returns the original string if it doesn't match the expected format or if an error occurs.

            Example:
                eval_tuple("(2*3, 5+2)") -> "(6,7)"

            Note:
                This function relies on the latex2sympy function which is assumed to be defined elsewhere in the code.
            """
            sl = s[1:-1].split(',')

            try:
                if s[0] == '(' and s[-1] == ')' and len(sl) > 1:
                    s = ','.join(
                        [
                            (
                                str(round(eval(str(latex2sympy(sub))), 2))
                                if 'infty' not in sub and sub not in ['a', '-a']
                                else sub
                            )
                            for sub in sl
                        ]
                    )
                    return f'({s})'

                elif s[0] == '[' and s[-1] == ']' and len(sl) > 1:
                    s = ','.join(
                        [
                            (
                                str(round(eval(str(latex2sympy(sub))), 2))
                                if 'infty' not in sub and sub not in ['a', '-a']
                                else sub
                            )
                            for sub in sl
                        ]
                    )
                    return f'[{s}]'

            except Exception:
                return s

            return s

        def is_equal(asw: str, gt_asw: str) -> bool:
            """
            Judge if `asw` is equivalent to `gt_asw`.

            This function checks if the given answers are equivalent, considering
            various scenarios such as tuples, lists separated by commas, and
            mathematical equivalence in LaTeX format.

            Args:
                asw (str): The answer string to be checked.
                gt_asw (str): The ground truth answer string to be matched against.

            Returns:
                bool: True if the answers are equivalent, otherwise False.

            """
            asw = asw.lower()
            gt_asw = gt_asw.lower()

            if asw.replace(' ', '') == '' or gt_asw.replace(' ', '') == '':
                return False

            if gt_asw.strip() == asw.strip():
                return True

            asw = eval_tuple(asw)
            gt_asw = eval_tuple(gt_asw)

            if gt_asw == asw:
                return True

            try:
                # Convert LaTeX format to a sympy expression and evaluate both expressions.
                # If the evaluated results are close enough (up to 2 decimal places), return True.
                if round(eval(str(latex2sympy(gt_asw))), 2) == round(
                    eval(str(latex2sympy(asw))), 2
                ):
                    return True
                else:
                    return False
            except:
                return False

        return is_equal(data_1, data_2)

    def __call__(self, data_1, data_2) -> bool:
        return self.apply(data_1, data_2)
