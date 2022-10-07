source env/bin/activate
cd examples/classifier_compression/
python early-exit-classifier.py --arch=resnet32_cifar_earlyexit --epoch=150 -b 128 --lr=0.1 --earlyexit_thresholds 0.8 0.8 --earlyexit_lossweights 0.4 0.3 -j 30 --out-dir . -n earlyexit .
python early-exit-classifier.py --arch=resnet32_cifar_hybrid --epoch=150 -b 128 --lr=0.1 --earlyexit_thresholds 0.8 0.8 --earlyexit_lossweights 0.4 0.3 -j 30 --out-dir . -n earlyexit .


python early-exit-classifier.py --arch=resnet32_cifar_earlyexit --epoch=150 -b 128 --lr=0.003 --earlyexit_thresholds 0.8 --earlyexit_lossweights 0.6 -j 30 --out-dir . -n earlyexit .

python early-exit-classifier.py --arch=resnet32_cifar_binary --epoch=150 -b 128 --lr=0.003 -j 30 --out-dir . -n earlyexit .

python early-exit-classifier.py --arch=resnet32_cifar_hybrid --epoch=150 -b 128 --lr=0.1 --earlyexit_thresholds 0.8 0.8 --earlyexit_lossweights 0.3 0.3 -j 30 --out-dir . -n earlyexit . --resume=/home/yinghan/Documents/distiller-master/examples/classifier_compression/earlyexit_hybrid/earlyexit_checkpoint.pth.tar

python early-exit-classifier.py --arch=resnet32_cifar_hybrid --epoch=150 -b 128 --lr=0.1 --earlyexit_thresholds 0.5 --earlyexit_lossweights 0.6 -j 30 --out-dir . -n earlyexit .

python early-exit-classifier-sep.py --arch=resnet32_cifar_hybrid --epoch=100 -b 128 --lr=0.1 --earlyexit_thresholds 0.8 0.8 0.8 --earlyexit_lossweights 0.3 0.3 0.3 -j 30 --out-dir . -n earlyexit . --resume=baseline-binary/earlyexit_checkpoint.pth.tar 

#full precision
python full-precision-classifier.py --arch=resnet32_cifar --epoch=150 -b 128 --lr=0.1 -j 30 --out-dir . -n earlyexit .


python early-exit-classifier.py --arch=resnet32_cifar_earlyexit --epoch=0 -b 128 --lr=0.1 --earlyexit_thresholds 1 1 --earlyexit_lossweights 0.3 0.3 -j 30 --out-dir . -n earlyexit . --resume
#cifar 100
python early-exit-classifier.py --arch=resnet32_cifar_hybrid --epoch=150 -b 128 --lr=0.1 --earlyexit_thresholds 0.8 --earlyexit_lossweights 0.6 -j 30 --out-dir . -n earlyexit .

python early-exit-classifier.py --arch=resnet32_cifar_hybrid --epoch=150 -b 128 --lr=0.1 --earlyexit_thresholds 0.4 0.4 0.8 0.8 --earlyexit_lossweights 0.15 0.15 0.15 0.15 -j 30 --out-dir . -n earlyexit .

#only ~20% samples exit early
#solution: increase weights on early exits OR make thresholds lower
python early-exit-classifier.py --arch=resnet32_cifar_hybrid --epoch=150 -b 128 --lr=0.1 --earlyexit_thresholds 1 --earlyexit_lossweights 0.4 -j 30 --out-dir . -n earlyexit .

python early-exit-classifier.py --arch=resnet32_cifar_hybrid --epoch=150 -b 128 --lr=0.1 --earlyexit_thresholds 0.8 0.8 --earlyexit_lossweights 0.3 0.4 -j 30 --out-dir . -n earlyexit .

python early-exit-classifier-train.py --arch=resnet32_cifar_hybrid --epoch=170 -b 128 --lr=0.1 --earlyexit_thresholds 0.8 0.8 0.8 0.8 --earlyexit_lossweights 0.1 0.1 0.1 0.1 -j 30 --out-dir . -n earlyexit . --resume=cifar100-10binary/earlyexit_checkpoint.pth.tar 

#train cnn
python early-exit-classifier-sep.py --arch=resnet32_cifar_hybrid --epoch=150 -b 128 --lr=0.1 --earlyexit_thresholds 1 --earlyexit_lossweights 0.4 -j 30 --out-dir . -n earlyexit . 
#train early exits
python early-exit-classifier-train.py --arch=resnet32_cifar_hybrid --epoch=170 -b 128 --lr=0.1 --earlyexit_thresholds 0.5 --earlyexit_lossweights 0.6 -j 30 --out-dir . -n earlyexit . --resume=cifar100-10binary/earlyexit_checkpoint.pth.tar 


#ANN-SNN
python ANN-SNN-classifier.py --arch=resnet32_cifar_SANN --epoch=100 -b 128 --lr=0.1 --earlyexit_thresholds 0.5 --earlyexit_lossweights 0.4 -j 30 --out-dir . -n earlyexit . --resume=cifar10-FP-earlyexit-20layer/earlyexit_best.pth.tar 

python full-precision-classifier.py --arch=resnet32_cifar_earlyexit --epoch=150 -b 128 --lr=0.1 --earlyexit_thresholds 1 --earlyexit_lossweights 0.4 -j 30 --out-dir . -n earlyexit . 
python full-precision-classifier.py --arch=resnet32_cifar_earlyexit --epoch=100 -b 128 --lr=0.1 --earlyexit_thresholds 1.0 --earlyexit_lossweights 0.4 -j 30 --out-dir . -n earlyexit . --resume=cifar10-FP-earlyexit-20layer/earlyexit_best.pth.tar --evaluate
python ANN-SNN-classifier.py --arch=resnet32_cifar_SANN --epoch=100 -b 128 --lr=0.1 --earlyexit_thresholds 1.5 --earlyexit_lossweights 0.4 -j 30 --out-dir . -n earlyexit . --resume=FP-no-batchnorm-10/earlyexit_best.pth.tar 

#edge-cloud collaboration
python early-exit-classifier-sep.py --arch=resnet32_cifar_binary --epoch=180 -b 128 --lr=0.1 -j 30 --out-dir . -n earlyexit . --resume=cifar100-binary/earlyexit_best.pth.tar --evaluate
python early-exit-classifier-train.py --arch=resnet32_cifar_binary --epoch=180 -b 128 --lr=0.1 -j 30 --out-dir . -n earlyexit .  --resume=cifar100-binary/earlyexit_best.pth.tar --earlyexit_lossweights 0.5 --earlyexit_thresholds 1.5
python early-exit-classifier-train.py --arch=resnet32_cifar_binary --epoch=180 -b 128 --lr=0.1 -j 30 --out-dir . -n earlyexit .  --resume=binary-exit-by-classifier/earlyexit_checkpoint.pth.tar --earlyexit_lossweights 0.5 --earlyexit_thresholds 1.5 --evaluate

#BlocTrain
python Block-Train.py --arch=resnet32_cifar_bloc1 --epoch=180 -b 128 --lr=0.02 -j 1 --out-dir . -n earlyexit . --earlyexit_lossweights 0.3 0.3 --earlyexit_thresholds 1.5 1.5 --deterministic

python Block-Train-hardclass.py --arch=resnet32_cifar_bloc3 --epoch=180 -b 128 --lr=0.02 -j 1 --out-dir . -n earlyexit . --earlyexit_lossweights 0.3 0.3 --earlyexit_thresholds 0.8 0.8 --deterministic --gpus=1 --resume=blocktrain-cifar100-50-Corrector/block3_checkpoint.pth.tar --evaluate

python Block-Train-ImageNet-pretrain.py --arch=resnet18_p --epoch=110 -b 256 --lr=0.01 -j 1 --out-dir . -n imagenet . --earlyexit_lossweights 0.3 --earlyexit_thresholds 0.8 --deterministic --gpus=1
#revision
python Block-Train-extend.py --arch=resnet32_cifar_extend --epoch=180 -b 128 --lr=0.1 -j 1 --out-dir . -n resnet32_extend . --earlyexit_lossweights 0.3 --earlyexit_thresholds 0.8 --gpus=0 --resume=resnet32_extend_concat/block3_best.pth.tar --evaluate

python Block-Train-hardclass.py --arch=resnet32_cifar_bloc3 --epoch=180 -b 128 --lr=0.1 -j 1 --out-dir . -n blocktrain . --earlyexit_lossweights 0.3 --earlyexit_thresholds 0.8 --gpus=1 --resume=blocktrain_resnet32_concat/block3_checkpoint.pth.tar --evaluate

python Block-Train-hardclass.py --arch=resnet32_cifar_bloc3 --epoch=180 -b 128 --lr=0.1 -j 1 --out-dir . -n blocktrain . --earlyexit_lossweights 0.3 --earlyexit_thresholds 0.8 --gpus=1 --resume=blocktrain_resnet32_b2/block2_checkpoint.pth.tar

