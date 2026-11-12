"""Generated from actual three-plan numeric proof; no inherited app/performance acceptance."""
READY = True
GLOBAL_STEP = 26523
VOICE_ID = '01yuewen'
SOURCE_SHA = 'a46bd6826d1074b93bc7a17379966d88005cd248136acd24aab4b6883e689bb0'
NATIVE_WEIGHTS_SHA = 'f33dcf83714f3c42cbdad5c85a6ec09d1e69bec29e6c173992bff429f72852f9'
PHASE_DSP_SHA = 'a89ad2f2658563cbcd17cb608306d13494623834f691e530a326cbcf2e76d073'
STYLE_FILE_SHA = '1c0c128370aa247bbf3eb77bf798c0f7d4d4583074f9d582e314c4a0570ca13c'
STYLE_RAW_SHA = '1aba56cb2955bc9940de8e3a9c68241acbfc79e126bc63b1386cd9953a05667f'
GRAPH_SHAS = {'encoder': '4b8dffbeaceb79655ed97088eaeac97fc50b2b3df737cba2580ae0774cdc8a15', 'acoustic': '5db4459a69fc160dee214fa151334ef070d1459e94256f4d6d1e45454ec29a2a', 'vocoder': 'adabfa70659495c2fc592e3cbe57851eeeb854c4332ce9b7bee29c16e6257995'}
PLAN_SHAS = {'encoder': '5aee1d8dd80ee3383d88151e1e70b9af9b387b1b5f325deff482b9578c8255ca', 'acoustic': '941ecb74b421a697168cafdd635983b243703435e804103ac785050bceb2528a', 'vocoder': '9f81b03e575bc5c0ffdfe830b210594d7ea6035c26651710cc92b5c61817a15c'}
EXPECTED_PROFILES = {'encoder': {'tokens': [[1, 3], [1, 128], [1, 512]], 'style': [[1, 256], [1, 256], [1, 256]], 'speed': [[1], [1], [1]]}, 'acoustic': {'d': [[1, 640, 3], [1, 640, 128], [1, 640, 512]], 't_en': [[1, 512, 3], [1, 512, 128], [1, 512, 512]], 'indices': [[3], [512], [832]], 'style': [[1, 256], [1, 256], [1, 256]]}, 'vocoder': {'features': [[1, 512, 6], [1, 512, 512], [1, 512, 1460]], 'style': [[1, 256], [1, 256], [1, 256]], 'f0': [[1, 6], [1, 512], [1, 1460]], 'source_phase': [[1, 1800, 9], [1, 153600, 9], [1, 438000, 9]], 'excitation_noise': [[1, 1800, 9], [1, 153600, 9], [1, 438000, 9]]}}
STREAM_POLICY = {'mode': 'natural_text_streaming', 'supported_F_range': [6, 1460], 'max_T': 730, 'split_strategy': 'whole_legal_utterance_then_lossless_semantic_fallback', 'speed': 1.0, 'preserve_full_text': True, 'allow_frame_truncation': False}
CHAIN_EVIDENCE_SHA = '94e9f8a1c0a1145d9d9a314b1d13472b0281d5364c11f423c87daac044191284'
OPERATING_F_RANGE = [6, 1460]
PHYSICAL_F_RANGE = [6, 1460]
TESTED_F_VALUES = [6, 1238, 1328, 1460]
