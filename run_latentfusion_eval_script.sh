python3 evaluate_pose.py --checkpoint-path /ssd_scratch/cvit/amoghtiwari/latentfusion/checkpoints/latentfusion-release.pth --dataset-dir /ssd_scratch/cvit/amoghtiwari/latentfusion/data/lm_format --dataset-type lm_format --num-input-views 8 --input-scene-dir /ssd_scratch/cvit/amoghtiwari/latentfusion/data/lm_format/train/000004/ --target-scene-dir /ssd_scratch/cvit/amoghtiwari/latentfusion/data/lm_format/test/000004/ --object-id 4 --out-dir /ssd_scratch/cvit/amoghtiwari/latentfusion/results --scene-name 000004 --base-name base --coarse-config configs/cross_entropy_linemod.toml --refine-config configs/adam_slow.toml


