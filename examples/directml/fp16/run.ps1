$modelPath = $args[0]
$escapedModelPath = $modelPath.ToString() | ConvertTo-Json

del .\cache -Recurse -Force -ErrorAction SilentlyContinue
(cat .\config_template.json -Raw).Replace("%%MODEL_PATH%%", $escapedModelPath) | out-file config.json -encoding ASCII
python -m olive.workflows.run --config config.json

$outSrcPath = dir cache *.onnx -recurse | select -first 1 -ExpandProperty FullName

$outDstFilename = [System.IO.Path]::GetFileNameWithoutExtension($modelPath) + "-fp16.onnx"
$outDstPath = $outDstFilename

move $outSrcPath $outDstPath