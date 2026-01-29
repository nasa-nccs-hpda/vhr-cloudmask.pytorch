


logger.info('Running CloudMaskPipeline.')
cMaskDirNum = toaDirNum + 1
cMaskDir = outDir / (str(cMaskDirNum) + '-masks')
cMaskDir.mkdir(exist_ok=True)
cMaskActualOutDir = cMaskDir / '5-toas'

cmpl = CloudMaskPipeline(output_dir=cMaskDir,
                            inference_regex_list=[str(toaFile)])

cmpl.predict()