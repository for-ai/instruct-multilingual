import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

import ctranslate2
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

app = FastAPI()
TRANSLATIONS = asyncio.Queue()
translator: ctranslate2.Translator = None


class NLLBRequest(BaseModel):
    source_language: str
    target_language: str
    texts: List[str]


class NLLBResponse(BaseModel):
    translated_texts: List[str]


@app.on_event("startup")
async def startup_event():
    global translator
    translator = ctranslate2.Translator(
        "models/nllb-200-3.3B-converted",
        device="auto",
        compute_type="float16",
        device_index=list(range(4)),
    )
    asyncio.create_task(batch_translate())


@app.post("/translate")
async def translate(data: NLLBRequest):
    translated_texts = await process_translation(
        data.source_language,
        data.target_language,
        data.texts,
    )

    return NLLBResponse(translated_texts=translated_texts)


async def process_translation(
    source_language: str,
    target_language: str,
    texts: List[str],
) -> List[str]:
    future = asyncio.get_running_loop().create_future()
    await TRANSLATIONS.put((source_language, target_language, texts, future))
    result = await future
    return result


async def batch_translate():
    while True:
        if not TRANSLATIONS.empty():
            processing_now = TRANSLATIONS.qsize()
            batch_src, batch_tgt, batch_text, batch_futures = [], [], [], []
            for _ in range(processing_now):
                src, tgt, text, future = await TRANSLATIONS.get()
                batch_src.append(src)
                batch_tgt.append(tgt)
                batch_text.append(text)
                batch_futures.append(future)

            results = await translate_batch_async(batch_src, batch_tgt, batch_text)
            for i, result in enumerate(results):
                batch_futures[i].set_result(result)

        await asyncio.sleep(0.1)


async def translate_batch_async(
    source_language: List[str],
    target_language: List[str],
    texts: List[List[str]],
) -> List[str]:
    tgt_lang_list = []
    tokenized = []

    for i, text_list in enumerate(texts):
        src_lang = source_language[i]
        tgt_lang = target_language[i]
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-3.3B",
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        tokenized.extend(
            [
                tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
                for text in text_list
            ]
        )
        tgt_lang_list.extend([[tgt_lang]] * len(text_list))

    def sync_func():
        return translator.translate_batch(
            tokenized,
            target_prefix=tgt_lang_list,
            max_batch_size=64,
        )

    with ThreadPoolExecutor() as executor:
        results = await asyncio.get_running_loop().run_in_executor(executor, sync_func)

    targets = [result.hypotheses[0][1:] for result in results]

    translations = [
        tokenizer.decode(tokenizer.convert_tokens_to_ids(target)) for target in targets
    ]

    rv: List[List[str]] = []

    for text in texts:
        pipi: List[str] = []
        for _ in text:
            pipi.append(translations.pop(0))
        rv.append(pipi)

    return rv


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
