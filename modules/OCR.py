import os
import cv2
import pytesseract
import base64
import json
import re
import openai
from PIL import Image
import sympy as sp
from modules.utils import make_client

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = 'YOUR_OPENAI_API_KEY'

client = make_client('')


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

example_image_path_1 = './data/exam_10.00.16.png' # 순환소수
example_image_path_2 = './data/exam_10.02.40.png' # 순환소수
example_image_path_3 = "./data/exam_9.41.54.png" # 지수
example_image_path_4 = './data/exam_9.42.01.png' # 루트
example_image_path_5 = './data/exam_10.03.14.png'# 지수
example_image_path_6 = './data/exam_9.51.32.png' # 특수기호
example_image_path_7 = './data/exam_9.52.21.png' # 특수기호
example_image_path_8 = './data/exam_9.52.46.png' # 특수기호
example_image_path_9 = './data/exam_9.45.48.png' # 선분
example_image_path_10 = './data/exam_10.11.44.png' # 선분
example_image_path_11 = './data/exam_9.52.55.png' # 선분
example_iamge_path_12 = './data/exam_9.58.15.png'# 비례식


encoded_example_image_1 = encode_image(example_image_path_1)
encoded_example_image_2 = encode_image(example_image_path_2)
encoded_example_image_3 = encode_image(example_image_path_3)
encoded_example_image_4 = encode_image(example_image_path_4)
encoded_example_image_5 = encode_image(example_image_path_5)
encoded_example_image_6 = encode_image(example_image_path_6)
encoded_example_image_7 = encode_image(example_image_path_7)
encoded_example_image_8 = encode_image(example_image_path_8)
encoded_example_image_9 = encode_image(example_image_path_9)
encoded_example_image_10 = encode_image(example_image_path_10)
encoded_example_image_11 = encode_image(example_image_path_11)
encoded_example_image_12 = encode_image(example_iamge_path_12)

# Few-shot example
img_system_prompt = """당신은 수학 문제가 담긴 이미지를 받으면 문제에 써진 텍스트를 Latex로 바꾸어서 보내면 됩니다.
형식은 다음과 같습니다.
[문제]
이미지의 문제를 latex로 변환한 것을 적으면 됩니다
이 때 한글로 적힌 문제는 적되, 만약 latex로 이루어진 수식이 있을 경우에는 [Latex_n]으로 표기해주세요.
"""

img_user_prompt = """
내가 보낸 수학 문제가 담긴 이미지에서 문제가 써진 텍스트를 LaTeX로 바꿔서 보여줘.
다음과 같은 사항을 주의해서 작성해줘


1. 순환소수가 있는 문제에서 소수점 아래 어디까지 순환소수 점이 있는지 확인하기
2. 절대 비례식을 분수로 나타내지 않기. 예를 들어 1:2를 frac{1}{2}로 나타내지 않기
3. 루트가 어디까지 있는지 확인하기
4. 극한이 있는 문제에서 좌극한과 우극한의 경우 극한의 부호까지 표시하기
5. 수열 문제에서 밑첨자를 확인하기
6. 부등식 문제에서 이하인지 미만인지, 이상인지 초과인지 확인하기
7. 지수 문제에서 지수인지 일반 문자인지 확인하기
8. 순환소수 문제에서 순환소수를 나타내는 점이 어디까지 있는지 확인한다. 이 때 순환소수는 선분과 헷갈리지 않도록 \\dot으로 표현한다.
"""

img_example = [
    {
        "role": "system",
        "content": img_system_prompt
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": img_user_prompt}
        ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_example_image_6}"}}]
    },
    {
        "role": "assistant",
        "content": """
        [문제]
        [Latex_1]의 양변에 [Latex_2]을 대입하면 [Latex_3].
        [Latex_4]의 양변을 [Latex_5]에 대해 미분하고 [Latex_6]을 대입하면 [Latex_7].

        Latex_1: \[
            \int_{1}^{x} f(t) dt = x^3 - ax + 1 \quad \cdots \quad  \text{㉠}
        \]
        Latex_2: \( x = 1 \)
        Latex_3: \( 1 - a + 1 = 0, \quad a = 2 \)
        Latex_4: \( \int_{1}^{x} f(t) dt = x^3 - ax + 1 \)
        Latex_5: \( x \)
        Latex_6: \( a = 2 \)
        Latex_7: \[
            f(x) = 3x^2 - 2 \text{이므로 } f(2) = 10
        \]
        """
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": img_user_prompt}
        ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_example_image_9}"}}]
    },
    {
        "role": "assistant",
        "content": """
        [문제]
        그림과 같이 중심이 [Latex_1], 반지름의 길이가 [Latex_2]이고 중심각의 크기가 [Latex_3]인 부채꼴 [Latex_4]이 있다.
        호 [Latex_5] 위에 점 [Latex_6], 선분 [Latex_7] 위에 점 [Latex_8], 선분 [Latex_9] 위에 점 [Latex_10]을 사각형 [Latex_11]이 되도록 잡는다.
        부채꼴 [Latex_12]의 내부에 점 [Latex_13]을 잡고, [Latex_14]이 되도록 잡고, 이등변삼각형 [Latex_15]에 색칠하여 얻은 그림을 [Latex_16]이라 하자.

        Latex_1: \( O \)
        Latex_2: 1
        Latex_3: \( \frac{\pi}{2} \)
        Latex_4: \( OA_1B_1 \)
        Latex_5: \( A_1B_1 \)
        Latex_6: \( P_1 \)
        Latex_7: \( OA_1 \)
        Latex_8: \( C_1 \)
        Latex_9: \( OB_1 \)
        Latex_10: \( D_1 \)
        Latex_11: \( OC_1P_1D_1 \)
        Latex_12: \( OA_1B_1 \)
        Latex_13: \( Q_1 \)
        Latex_14: \( \overline{P_1Q_1} = \overline{A_1Q_1}, \quad \angle P_1Q_1A_1 = \frac{\pi}{2} \)
        Latex_15: \( P_1Q_1A_1 \)
        Latex_16: \( R_1 \)
        """
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": img_user_prompt}
        ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_example_image_10}"}}]
    },
    {
        "role": "assistant",
        "content": """
        [문제]
        [Latex_1]에 [Latex_2]을 대입하면 [Latex_3].
        따라서 [Latex_4]
        또 [Latex_5]에 [Latex_6]을 대입하면 [Latex_7].

        Latex_1: \[ y = -x^2 + x + 2 \]
        Latex_2: \( y = 0 \)
        Latex_3: \[
            -x^2 + x + 2 = 0, \quad x^2 - x - 2 = 0, \quad (x + 1)(x - 2) = 0, \quad x = -1 \text{ 또는 } x = 2
        \]
        Latex_4: \( A(-1, 0), \quad B(2, 0) \)
        Latex_5: \( y = -x^2 + x + 2 \)
        Latex_6: \( x = 0 \)
        Latex_7: \( y = 2, \quad C(0, 2) \)
        """
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": img_user_prompt}
        ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_example_image_11}"}}]
    },
    {
        "role": "assistant",
        "content": """
        [문제]
        [Latex_1]에서 [Latex_2] 또는 [Latex_3]이다.
        [Latex_4]에서 [Latex_5].
        [Latex_6]에서 방정식 [Latex_7]의 해가 존재하므로 [Latex_8]이고 [Latex_9].

        Latex_1: \( g(a\pi) = -1 \text{ 또는 } g(a\pi) = 1 \)
        Latex_2: \( g(a\pi) = -1 \)
        Latex_3: \( g(a\pi) = 1 \)
        Latex_4: \( \sin(a\pi) = -1 \text{에서 } a = \frac{3}{2}, \sin(a\pi) = 1 \text{에서 } a = \frac{1}{2} \)
        Latex_5: \( a = \frac{3}{2} \text{ 또는 } a = \frac{1}{2} \)
        Latex_6: (나)
        Latex_7: \( f(g(x)) = 0 \)
        Latex_8: \( -1 \leq t \leq 1 \)
        Latex_9: \( f(t) = 0 \text{인 실수 } t \text{가 존재한다.} \)
        """
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": img_user_prompt}
        ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_example_image_12}"}}]
    },
    {
        "role": "assistant",
        "content": """
        [문제]
        [Latex_1]이므로 [Latex_2], [Latex_3]이고 [Latex_4].

        Latex_1: \[
            \text{원 } C \text{의 반지름의 길이가 } 3 \text{이므로}
        \]
        Latex_2: \( \overline{AA'} = 3, \quad \overline{AP} = 3 \)
        Latex_3: \[
            \text{직선 } l \text{의 기울기가 } -\frac{4}{3} \text{이므로}
        \]
        Latex_4: \[
            \frac{\overline{BA'}}{\overline{AA'}} = \frac{4}{3}, \quad \text{즉 } \overline{BA'} = 4, \quad \overline{BA} = 5, \quad \overline{BP} = 8
        \]
        """
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": img_user_prompt}
        ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_example_image_5}"}}]
    },
    {
        "role": "assistant",
        "content":
        """
        [문제]
        [Latex_1]일 때, 상수 [Latex_2]에 대하여 [Latex_3]의 값을 구하시오.

        Latex_1: \[
            \left( \frac{az^2}{x^3 y^b} \right)^3 = -\frac{8z^6}{x^c y^9}
        \]
        Latex_2: \( a, b, c \)
        Latex_3: \( a + b - c \)
        """
    },
    {
        "role": "assistant",
        "content": """
        [문제]
        어떤 기약분수를 순환소수로 나타내는데, 예준이는 분모를 잘못 보아 [Latex_1]이 되었고, 연서는 분자를 잘못 보아 [Latex_2]이 되었다. 이 때 처음 기약분수를 순환소수로 나타내면?

        [선지]
        1. [Latex_3]
        2. [Latex_4]
        3. [Latex_5]
        4. [Latex_6]
        5. [Latex_7]

        Latex_1: \(1.\\dot{1}\)
        Latex_2: \(1.\\dot{1}\\dot{3}\)
        Latex_3: \(0.1\\dot{1}\)
        Latex_4: \(0.\\dot{1}\\dot{0}\)             
        Latex_5: \(0.\\dot{1}0\\dot{1}\)
        Latex_6: \(0.0\\dot{1}\)
        Latex_7: \(0.\\dot{0}\\dot{1}\)
        """
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": img_user_prompt}
        ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_example_image_2}"}}]
    },
    {
        "role": "assistant",
        "content": """
        [문제]
        [Latex_1]에서 [Latex_2] 안에 알맞은 수는 무엇인가?

        [선지]
        1. [Latex_3]
        2. [Latex_4]
        3. [Latex_5]
        4. [Latex_6]
        5. [Latex_7]

        Latex_1: \[
            0.\\dot{5}2\\dot{3} = 523 \\times \\boxed{\\phantom{0}}
        \]
        Latex_2: \( \\boxed{\\phantom{0}} \)
        Latex_3: \(0.1\)
        Latex_4: \(0.\\dot{1}\)
        Latex_5: \(0.\\dot{0}\\dot{1}\)
        Latex_6: \(0.\\dot{0}0\\dot{1}\)
        Latex_7: \(0.0\\dot{0}\\dot{1}\)
        """
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": img_user_prompt}
        ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_example_image_4}"}}]
    },
    {
        "role": "assistant",
        "content": """
        [문제]
        [Latex_1]의 값은 무엇인가?

        [선지]
        1. [Latex_2]
        2. [Latex_3]
        3. [Latex_4]
        4. [Latex_5]
        5. [Latex_6]

        Latex_1: \[
            \left( \frac{4}{2^{\\sqrt{2}}} \right)^{2 + \\sqrt{2}}
        \]
        Latex_2: \(\\frac{1}{4}\)
        Latex_3: \(\\frac{1}{2}\)
        Latex_4: 1
        Latex_5: 2
        Latex_6: 4
        """
    },
]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    return binary

def extract_text_with_ocr(image_path):
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image_pil = Image.fromarray(preprocessed_image)

      # Tesseract 설정
    config_kor = r'--oem 3 --psm 6'  # 한국어
    config_math = r'--oem 3 --psm 6'  # 수식 인식

      # 한국어와 수학 수식을 별도로 인식
    text_kor = pytesseract.image_to_string(preprocessed_image_pil, config=config_kor, lang='kor')
    text_math = pytesseract.image_to_string(preprocessed_image_pil, config=config_math)

    return text_kor, text_math

def parse_equations_with_sympy(text):
      equations = text.split('\n')
      latex_equations = []
      for eq in equations:
          try:
              sympy_eq = sp.sympify(eq)
              latex_eq = sp.latex(sympy_eq)
              latex_equations.append(latex_eq)
          except:
              latex_equations.append(eq)  # SymPy로 파싱하지 못한 경우 원래 텍스트 사용
      return latex_equations

def extract_latex_and_text(text):
      # \text{} 패턴을 우선적으로 처리하여 한글만 남기기
      text_pattern = re.compile(r'\\text\{(.+?)\}')
      text_replaced = text_pattern.sub(lambda m: m.group(1), text)  # \text{} 내용만 한글로 남김

      # LaTeX 수식 패턴 정의
      latex_pattern = re.compile(r'(\\\(.+?\\\))|(\\\[.+?\\\])')

      # Placeholder list to store LaTeX parts
      latex_expressions = []
      match_index = 1

      # LaTeX 수식을 [Latex_n]으로 대체하여 처리된 텍스트 생성
      def replace_with_placeholder(match):
          nonlocal match_index
          placeholder = f'[Latex_{match_index}]'
          latex_expressions.append(match.group())
          match_index += 1
          return placeholder

      processed_text = latex_pattern.sub(replace_with_placeholder, text_replaced)

      return processed_text, latex_expressions
# OCR 메인 함수 

def OCR_image(img_name):
    encoded_image = encode_image(img_name)
    text_kor, text_math = extract_text_with_ocr(img_name)
    latex_math = parse_equations_with_sympy(text_math)

      # GPT-4o 모델 메시지 구성
    img_messages = [
        {
            "role": "system",
            "content": img_system_prompt
        },
        {
            "role": "user",
            "content": img_user_prompt + "\n\nOCR로 추출된 텍스트:\n" + text_kor + '\n\n' + ' \\ '.join(latex_math)
        },
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}]
        }
    ]
    
    # Make the API call

    client = make_client("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")

    img_response = client.chat.completions.create(
        model="gpt-4o",
        messages=img_example + img_messages,
        temperature=0,
        max_tokens=2048,
        top_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    result = img_response.choices[0].message.content.strip()
    question_text, latex_parts = extract_latex_and_text(result)

    # 결과를 원하는 형식으로 구성
    formatted_result = {
        "File": img_name,
        "Question": question_text
    }

    # LaTeX 수식을 `Latex_n` 형식으로 추가
    for i, latex in enumerate(latex_parts):
        formatted_result[f"Latex_{i + 1}"] = latex.strip()

    return formatted_result
