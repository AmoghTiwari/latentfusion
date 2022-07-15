MP_FILE_NAME="moped.tgz"
mkdir -p /ssd_scratch/cvit/amoghtiwari/latentfusion/
cd /ssd_scratch/cvit/amoghtiwari/latentfusion

mkdir checkpoints
scp amoghtiwari@ada.iiit.ac.in:/share3/amoghtiwari/checkpoints/latentfusion_checkpoints/latentfusion-release.pth checkpoints/

mkdir data
cd data
scp amoghtiwari@ada.iiit.ac.in:/share3/amoghtiwari/data/$MP_FILE_NAME ./
tar -xvzf $MP_FILE_NAME
rm $MP_FILE_NAME

