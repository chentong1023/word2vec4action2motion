import models.motion_vae as vae_models
from trainer.vae_trainer import *
from utils.plot_script import *
import utils.paramUtil as paramUtil
from utils.utils_ import *
from options.evaluate_vae_options import *
from dataProcessing import dataset
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertModel
from embed.word_embeding import sequence_embedding
from models.mlp import MLP

if __name__ == "__main__":
    parser = TestOptions()
    opt = parser.parse()
    joints_num = 0
    input_size = 72
    data = None
    label_dec = None
    dim_category = 31
    enumerator = None
    device = torch.device("cuda:" + str(opt.gpu_id))

    opt.dim_bedding = 768
    opt.save_root = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name)
    opt.model_path = os.path.join(opt.save_root, "model")
    opt.joints_path = os.path.join(opt.save_root, "joints")

    model_file_path = os.path.join(opt.model_path, opt.which_epoch + ".tar")
    result_path = os.path.join(
        opt.result_path, opt.dataset_type, opt.name + opt.name_ext
    )

    if opt.dataset_type == "humanact12":
        dataset_path = "./dataset/humanact12"
        input_size = 72
        joints_num = 24
        label_dec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        raw_offsets = paramUtil.humanact12_raw_offsets
        kinematic_chain = paramUtil.humanact12_kinematic_chain
        enumerator = paramUtil.humanact12_coarse_action_enumerator

    elif opt.dataset_type == "mocap":
        dataset_path = "./dataset/mocap/mocap_3djoints/"
        clip_path = "./dataset/mocap/pose_clip.csv"
        input_size = 60
        joints_num = 20
        raw_offsets = paramUtil.mocap_raw_offsets
        kinematic_chain = paramUtil.mocap_kinematic_chain
        label_dec = [0, 1, 2, 3, 4, 5, 6, 7]
        enumerator = paramUtil.mocap_action_enumerator

    elif opt.dataset_type == "ntu_rgbd_vibe":
        file_prefix = "./dataset"
        motion_desc_file = "ntu_vibe_list.txt"
        joints_num = 18
        input_size = 54
        label_dec = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
        labels = paramUtil.ntu_action_labels
        enumerator = paramUtil.ntu_action_enumerator
        raw_offsets = paramUtil.vibe_raw_offsets
        kinematic_chain = paramUtil.vibe_kinematic_chain
    else:
        raise NotImplementedError("This dataset is unregonized!!!")

    opt.dim_category = len(label_dec)
    opt.dim_embedding = 768

    action_embed_dict = []
    action_dict = enumerator
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained(
        "bert-base-uncased", output_hidden_states=True
    )
    bert_model.eval()
    for key, value in action_dict.items():
        action_embed_dict.append(sequence_embedding(value, bert_tokenizer, bert_model))

    opt.pose_dim = input_size

    if opt.time_counter:
        opt.input_size = input_size + opt.dim_bedding + 1
    else:
        opt.input_size = input_size + opt.dim_bedding

    opt.output_size = input_size

    model = torch.load(model_file_path)
    prior_net = vae_models.GaussianGRU(
        opt.input_size,
        opt.dim_z,
        opt.hidden_size,
        opt.prior_hidden_layers,
        opt.num_samples,
        device,
    )
    project_net = MLP(768, opt.dim_embedding, [128, 64])
    if opt.use_lie:
        decoder = vae_models.DecoderGRULie(
            opt.input_size + opt.dim_z,
            opt.output_size,
            opt.hidden_size,
            opt.decoder_hidden_layers,
            opt.num_samples,
            device,
        )
    else:
        decoder = vae_models.DecoderGRU(
            opt.input_size + opt.dim_z,
            opt.output_size,
            opt.hidden_size,
            opt.decoder_hidden_layers,
            opt.num_samples,
            device,
        )

    prior_net.load_state_dict(model["prior_net"])
    decoder.load_state_dict(model["decoder"])
    project_net.load_state_dict(model["project_net"])
    prior_net.to(device)
    decoder.to(device)
    project_net.to(device)
    if opt.use_lie:
        if opt.dataset_type == "humanact12":
            data = dataset.MotionFolderDatasetHumanAct12(
                dataset_path, opt, lie_enforce=opt.lie_enforce
            )
        elif opt.dataset_type == "ntu_rgbd_vibe":
            data = dataset.MotionFolderDatasetNtuVIBE(
                file_prefix,
                motion_desc_file,
                labels,
                opt,
                joints_num=joints_num,
                do_offset=True,
                extract_joints=paramUtil.kinect_vibe_extract_joints,
            )
        elif opt.dataset_type == "mocap":
            data = dataset.MotionFolderDatasetMocap(clip_path, dataset_path, opt)
        motion_dataset = dataset.MotionDataset(data, opt)
        motion_loader = DataLoader(
            motion_dataset,
            batch_size=opt.batch_size,
            drop_last=True,
            num_workers=2,
            shuffle=True,
        )
        trainer = TrainerLie(
            motion_loader, action_embed_dict, opt, device, raw_offsets, kinematic_chain
        )
    else:
        trainer = Trainer(None, action_embed_dict, opt, device)

    if opt.do_random:
        fake_motion, classes = trainer.evaluate(prior_net, project_net, decoder, opt.num_samples)
        fake_motion = fake_motion.cpu().numpy()
    elif opt.eval_type != "":
        category_em = sequence_embedding(opt.eval_type, bert_tokenizer, bert_model)

        # category_em1 = sequence_embedding("run", bert_tokenizer, bert_model)
        # category_em2 = sequence_embedding("walk", bert_tokenizer, bert_model)
        # category_em = (category_em1 - category_em2) * 2 + category_em1

        categories = np.stack([category_em for i in range(opt.replic_times)])
        categories_em = torch.from_numpy(categories).to(device).requires_grad_(False)
        fake_motion, _ = trainer.evaluate(
            prior_net, project_net, decoder, opt.replic_times, categories_em
        )
        fake_motion = fake_motion.cpu().numpy()
    else:
        categories = np.arange(opt.dim_category).repeat(opt.replic_times, axis=0)
        num_samples = categories.shape[0]
        category_em, classes = trainer.get_cate_word_embedding(categories)
        fake_motion, _ = trainer.evaluate(prior_net, project_net, decoder, num_samples, category_em)
        fake_motion = fake_motion.cpu().numpy()

    print(fake_motion.shape)
    for i in range(fake_motion.shape[0]):
        if opt.eval_type != "":
            class_type = opt.eval_type
        else:
            class_type = enumerator[label_dec[classes[i]]]

        motion_orig = fake_motion[i]
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        keypoint_path = os.path.join(result_path, "keypoint")
        if not os.path.exists(keypoint_path):
            os.makedirs(keypoint_path)
        file_name = os.path.join(result_path, class_type + str(i) + ".gif")
        offset = np.matlib.repmat(
            np.array([motion_orig[0, 0], motion_orig[0, 1], motion_orig[0, 2]]),
            motion_orig.shape[0],
            joints_num,
        )
        file_name2 = os.path.join(result_path, class_type + str(i) + "_traj.png")

        motion_mat = motion_orig - offset

        motion_mat = motion_mat.reshape(-1, joints_num, 3)
        np.save(
            os.path.join(keypoint_path, class_type + str(i) + "_3d.npy"), motion_mat
        )

        if opt.dataset_type == "humanact12":
            plot_3d_motion_v2(
                motion_mat, kinematic_chain, save_path=file_name, interval=80
            )
            plot_3d_motion_with_trajec(
                motion_mat, kinematic_chain,
                save_path=file_name2, interval=80
            )

        elif opt.dataset_type == "ntu_rgbd_vibe":
            plot_3d_motion_v2(
                motion_mat, kinematic_chain, save_path=file_name, interval=80
            )

        elif opt.dataset_type == "mocap":
            plot_3d_motion_v2(
                motion_mat,
                kinematic_chain,
                save_path=file_name,
                interval=80,
                dataset="mocap",
            )
