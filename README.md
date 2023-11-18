# face_liveness_detection
Treinamento de uma Rede Neural Convolucional (CNN, Resnet18) para detecção de vivacidade facial em imagens.

### 1) Baixar o dataset OULU-NPU (faces detectadas, alinhadas e cropadas em 224x224)
![OULU-NPU samples](https://github.com/biesseck/face_liveness_detection/blob/main/oulu-npu_samples.png)
[https://drive.google.com/drive/folders/1UfUO_IbeDFVq7oHjMLBnMr5zjB6UaVqR?usp=sharing](https://drive.google.com/drive/folders/1UfUO_IbeDFVq7oHjMLBnMr5zjB6UaVqR?usp=sharing)
```
unzip oulu-npu_frames_crop224x224.zip    # extrai as imagens
```

### 2) Criar e ativar um conda environment (via terminal de comandos)
```
conda create -n face_liveness_detection python=3.10
conda activate face_liveness_detection

# SEU TERMINAL DEVERÁ FICAR COM ESTA APARÊNCIA
# (face_liveness_detection) <username>@<computer_name>:~$
```

### 3) Clonar este repositório (via terminal de comandos)
```
git clone https://github.com/biesseck/face_liveness_detection.git
```

### 4) Instalar bibliotecas Python (via terminal de comandos)
```
cd face_liveness_detection          # entra no diretório do projeto
pip3 install -r requirements.txt    # instala as dependências python
```

### 5) Configurar o path do dataset OULU-NPU que você baixou
* Abra o arquivo `configs/oulu-npu_frames_r18.py`
* Modifique o path na linha `config.dataset_path = '/datasets_ufpr/liveness/oulu-npu_frames_crop224x224'`

### 6) Treinar a CNN Resnet18
```
export WORLD_SIZE=1; export RANK=0; python train_resnet.py configs/oulu-npu_frames_r18.py
```

