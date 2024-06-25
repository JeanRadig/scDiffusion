cd VAE
echo "Training Autoencoder, this might take a long time"
CUDA_VISIBLE_DEVICES=0 python VAE_train.py --data_dir 'path/to/anndata.h5ad' --num_genes 18996 --save_dir 'path/to/saved_VAE_model' --max_steps 200000
echo "Training Autoencoder done"

cd ..
echo "Training diffusion backbone"
CUDA_VISIBLE_DEVICES=0 python cell_train.py --data_dir 'path/to/anndata.h5ad'  --vae_path 'path/to/saved_VAE_model/VAE_checkpoint.pt' \
    --save_dir 'path/to/saved_diffusion_model' --model_name 'my_diffusion' --save_interval 20000
echo "Training diffusion backbone done"

echo "Training classifier"
CUDA_VISIBLE_DEVICES=0 python classifier_train.py --data_dir 'path/to/anndata.h5ad' --model_path "path/to/saved_classifier_model" \
    --iterations 40000 --vae_path 'path/to/saved_VAE_model/VAE_checkpoint.pt'
echo "Training classifier, done"