conda activate ScientificResearch

$env:PYTHONPATH = "D:\ScientificResearch\DINO-SAM-GECO2\src\models\backbones"


cd D:\ScientificResearch\DINO-SAM-GECO2\src\models
python -m backbones.dinov3_adapter
python -m backbones.sam3_adapter