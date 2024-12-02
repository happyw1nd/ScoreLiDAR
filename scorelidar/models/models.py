import torch
import torch.nn as nn
import torch.nn.functional as F
import scorelidar.models.minkunet as minknet
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d
from scorelidar.utils.scheduling import beta_func
from tqdm import tqdm
from os import makedirs, path

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from scorelidar.utils.collations import *
from scorelidar.utils.metrics import ChamferDistance, PrecisionRecall
from diffusers import DPMSolverMultistepScheduler


#######################################
# Modules
#######################################

class ScoreLiDAR(LightningModule):
    def __init__(self, diff_path, cfg):
        super().__init__()
        ckpt_diff = torch.load(diff_path)
        self.cfg = cfg

        from copy import deepcopy

        state_dict = ckpt_diff['state_dict']
        new_state_dict_diff = {}
        new_state_dict_enc = {}
        for key in state_dict:
            new_key = key.replace('model.', '')  # delete 'model.' prefix to fit pytorch model structures
            if 'partial_enc' in new_key:
                new_key = key.replace('partial_enc.', '')
                new_state_dict_enc[new_key] = (state_dict[key])
            else:
                new_state_dict_diff[new_key] = (state_dict[key])

        self.save_hyperparameters(ckpt_diff['hyper_parameters'])

        self.partial_enc = minknet.MinkGlobalEnc(in_channels=3, out_channels=self.hparams['model']['out_dim'])
        self.partial_enc.load_state_dict(new_state_dict_enc)

        new_state_dict_diff_g = deepcopy(new_state_dict_diff)
        self.generator = minknet.MinkUNetDiff(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()
        self.generator.load_state_dict(new_state_dict_diff_g)

        new_state_dict_diff_t = deepcopy(new_state_dict_diff)
        self.teacher = minknet.MinkUNetDiff(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()
        self.teacher.load_state_dict(new_state_dict_diff_t)

        new_state_dict_diff_s = deepcopy(new_state_dict_diff)
        self.auxDiff = minknet.MinkUNetDiff(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()
        self.auxDiff.load_state_dict(new_state_dict_diff_s)        

        # alphas and betas
        if self.hparams['diff']['beta_func'] == 'cosine':
            self.betas = beta_func[self.hparams['diff']['beta_func']](self.hparams['diff']['t_steps'])
        else:
            self.betas = beta_func[self.hparams['diff']['beta_func']](
                    self.hparams['diff']['t_steps'],
                    self.hparams['diff']['beta_start'],
                    self.hparams['diff']['beta_end'],
            )

        self.t_steps = self.hparams['diff']['t_steps']
        self.s_steps = 1
        self.s_steps_val = 8
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=torch.device('cuda')
        )

        self.alphas_cumprod_prev = torch.tensor(
            np.append(1., self.alphas_cumprod[:-1].cpu().numpy()), dtype=torch.float32, device=torch.device('cuda')
        )

        self.betas = torch.tensor(self.betas, device=torch.device('cuda'))
        self.alphas = torch.tensor(self.alphas, device=torch.device('cuda'))

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod) 
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)
        self.posterior_log_var = torch.log(
            torch.max(self.posterior_variance, 1e-20 * torch.ones_like(self.posterior_variance))
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        # for fast sampling
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(num_inference_steps=self.s_steps)

        self.dpm_scheduler_val = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler_val.set_timesteps(num_inference_steps=self.s_steps_val)
        self.scheduler_to_cuda()

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.hparams['data']['resolution'],2*self.hparams['data']['resolution'],100)

        self.w_uncond = cfg['train']['uncond_w']

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

        self.dpm_scheduler_val.timesteps = self.dpm_scheduler_val.timesteps.cuda()
        self.dpm_scheduler_val.betas = self.dpm_scheduler_val.betas.cuda()
        self.dpm_scheduler_val.alphas = self.dpm_scheduler_val.alphas.cuda()
        self.dpm_scheduler_val.alphas_cumprod = self.dpm_scheduler_val.alphas_cumprod.cuda()
        self.dpm_scheduler_val.alpha_t = self.dpm_scheduler_val.alpha_t.cuda()
        self.dpm_scheduler_val.sigma_t = self.dpm_scheduler_val.sigma_t.cuda()
        self.dpm_scheduler_val.lambda_t = self.dpm_scheduler_val.lambda_t.cuda()
        self.dpm_scheduler_val.sigmas = self.dpm_scheduler_val.sigmas.cuda()

    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:,None,None].cuda() * x + \
                self.sqrt_one_minus_alphas_cumprod[t][:,None,None].cuda() * noise

    def classfree_forward_generator(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward_generator(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward_generator(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def visualize_step_t(self, x_t, gt_pts, pcd, pcd_mean, pcd_std, pidx=0):
        points = x_t.F.detach().cpu().numpy()
        points = points.reshape(gt_pts.shape[0],-1,3)
        obj_mean = pcd_mean[pidx][0].detach().cpu().numpy()
        points = np.concatenate((points[pidx], gt_pts[pidx]), axis=0)

        dist_pts = np.sqrt(np.sum((points - obj_mean)**2, axis=-1))
        dist_idx = dist_pts < self.hparams['data']['max_range']

        full_pcd = len(points) - len(gt_pts[pidx])
        print(f'\n[{dist_idx.sum() - full_pcd}|{dist_idx.shape[0] - full_pcd }] points inside margin...')

        pcd.points = o3d.utility.Vector3dVector(points[dist_idx])
       
        colors = np.ones((len(points), 3)) * .5
        colors[:len(gt_pts[0])] = [1.,.3,.3]
        colors[-len(gt_pts[0]):] = [.3,1.,.3]
        pcd.colors = o3d.utility.Vector3dVector(colors[dist_idx])

    def reset_partial_pcd(self, x_part, x_uncond, x_mean, x_std):
        x_part = self.points_to_tensor(x_part.F.reshape(x_mean.shape[0],-1,3).detach(), x_mean, x_std)
        x_uncond = self.points_to_tensor(
                torch.zeros_like(x_part.F.reshape(x_mean.shape[0],-1,3)), torch.zeros_like(x_mean), torch.zeros_like(x_std)
        )

        return x_part, x_uncond
    
    def reset_partial_pcd_part(self, x_part, x_mean, x_std):
        x_part = self.points_to_tensor(x_part.F.reshape(x_mean.shape[0],-1,3).detach(), x_mean, x_std)

        return x_part

    def p_sample_loop(self, x_init, x_t, x_cond, x_uncond, gt_pts, x_mean, x_std):
        pcd = o3d.geometry.PointCloud()
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = torch.ones(gt_pts.shape[0]).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()
            
            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

            # this is needed otherwise minkEngine will keep "stacking" coords maps over the x_part and x_uncond
            # i.e. memory leak
            x_cond = self.reset_partial_pcd_part(x_cond, x_mean, x_std)
            torch.cuda.empty_cache()

        makedirs(f'{self.logger.log_dir}/generated_pcd/', exist_ok=True)

        return x_t
    
    def p_sample_one_step(self, x_init, x_t, x_cond, x_mean, x_std):
        pcd = o3d.geometry.PointCloud()
        self.scheduler_to_cuda()

        assert len(self.dpm_scheduler.timesteps) == self.s_steps
        for t in range(len(self.dpm_scheduler.timesteps)):
            t = torch.full((x_init.shape[0],), self.dpm_scheduler.timesteps[t]).cuda()
            
            noise_t = self.forward_generator(x_t, x_t.sparse(), x_cond, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

        makedirs(f'{self.logger.log_dir}/generated_pcd/', exist_ok=True)

        x_cond = self.reset_partial_pcd_part(x_cond, x_mean, x_std)
        torch.cuda.empty_cache()

        return x_t
    
    def p_sample_one_step_val(self, x_init, x_t, x_cond, x_uncond, x_mean, x_std):
        pcd = o3d.geometry.PointCloud()
        self.scheduler_to_cuda()

        assert len(self.dpm_scheduler_val.timesteps) == self.s_steps_val
        for t in range(len(self.dpm_scheduler_val.timesteps)):
            t = torch.full((x_init.shape[0],), self.dpm_scheduler_val.timesteps[t]).cuda()
            
            noise_t = self.classfree_forward_generator(x_t, x_cond, x_uncond, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler_val.step(noise_t, t[0], input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

        makedirs(f'{self.logger.log_dir}/generated_pcd/', exist_ok=True)

        x_cond = self.reset_partial_pcd_part(x_cond, x_mean, x_std)
        torch.cuda.empty_cache()

        return x_t

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

    def forward_aux(self, x_full, x_full_sparse, x_part, t):
        part_feat = self.partial_enc(x_part)
        out = self.auxDiff(x_full, x_full_sparse, part_feat, t)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)
    
    def forward_generator(self, x_full, x_full_sparse, x_part, t):
        part_feat = self.partial_enc(x_part)
        out = self.generator(x_full, x_full_sparse, part_feat, t)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)
    
    def forward_teacher(self, x_full, x_full_sparse, x_part, t):
        part_feat = self.partial_enc(x_part)
        out = self.teacher(x_full, x_full_sparse, part_feat, t)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)

    def points_to_tensor(self, x_feats, mean, std):
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord[:,1:] = feats_to_coord(x_feats[:,1:], self.hparams['data']['resolution'], mean, std)

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def point_set_to_sparse(self, p_full, n_part):
        p_full = p_full[0].cpu().detach().numpy()
    
        dist_part = np.sum(p_full**2, -1)**.5
        p_full = p_full[(dist_part < 50.) & (dist_part > 3.5)]
        p_full = p_full[p_full[:,2] > -4.]

        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(p_full)

        pcd_part = pcd_part.farthest_point_down_sample(n_part)
        p_part = torch.tensor(np.array(pcd_part.points))


        return p_part
    
    def compute_curvature(self, points, k=180):
        batch_size, point_num, _ = points.shape

        distance_matrix = torch.cdist(points, points, p=2)
        knn_indices = torch.argsort(distance_matrix, dim=-1)[:, :, 1:k+1]

        curvatures = []
        for b in range(batch_size):
            curv_batch = []
            for i in range(point_num):
                neighbors = points[b][knn_indices[b, i]] 
                centroid = neighbors.mean(dim=0)  
                cov_matrix = ((neighbors - centroid).T @ (neighbors - centroid)) / k 
                eigenvalues = torch.linalg.eigvalsh(cov_matrix)
                curvature = eigenvalues.min() / eigenvalues.sum() 
                curv_batch.append(curvature)
            curvatures.append(torch.tensor(curv_batch))

        curvatures = torch.stack(curvatures)
        return curvatures  

    def select_keypoints(self, points, curvatures, num_keypoints):
        batch_size, point_num, _ = points.shape
        keypoints = []
        indices = []
        for b in range(batch_size):
            curvature_indices = torch.argsort(curvatures[b], descending=True)[:num_keypoints]
            keypoints.append(points[b][curvature_indices])
            indices.extend((b * point_num + curvature_indices).tolist())
        keypoints = torch.stack(keypoints)
        indices = torch.tensor(indices, dtype=torch.long)
        return keypoints, indices
    
    def training_step(self, batch:dict, batch_idx, optimizer_idx):
        if optimizer_idx == 0: # train simulator

            # for classifier-free guidance switch between conditional and unconditional training
            cond = torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['pcd_full'].shape[0] == 1 
            if cond:
                x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            else:
                x_part = self.points_to_tensor(
                    torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
                )
            
            # generate a batch of samples from Generator
            with torch.no_grad():
                self.generator.eval()
                t_sample = batch['pcd_part'].repeat(1,10,1) + torch.randn(batch['pcd_part'].repeat(1,10,1).shape, device=self.device)
                x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std']) # to be denoised by Generator
                generated_sample = self.p_sample_one_step(t_sample, x_full, x_part, batch['mean'], batch['std'])
                generated_sample = generated_sample.F.reshape(t_sample.shape[0],-1,3)

            # add noise to generated samples
            with torch.no_grad():
                t = torch.randint(0, self.t_steps, size=(generated_sample.shape[0],)).cuda()
                noise = torch.randn(batch['pcd_full'].shape, device=self.device)
                t_sample = generated_sample + self.q_sample(torch.zeros_like(generated_sample), t, noise)
                x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std'])

            # denoise generated samples with auxDiff
            self.auxDiff.train()
            noise_simulator = self.forward_aux(x_full, x_full.sparse(), x_part, t)

            # calculate loss
            simulator_loss = self.p_losses(noise, noise_simulator)

            # log info on progress bar
            self.log('simulator_loss', simulator_loss, prog_bar=True)
            torch.cuda.empty_cache()

            return simulator_loss
        
        if optimizer_idx == 1: # train generator

            # for classifier-free guidance switch between conditional and unconditional training
            cond = torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['pcd_full'].shape[0] == 1
            if cond:
                x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            else:
                x_part = self.points_to_tensor(
                    torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
                )

            # get a batch of sample from Generator
            self.generator.train()
            t_sample = batch['pcd_part'].repeat(1,10,1) + torch.randn(batch['pcd_part'].repeat(1,10,1).shape, device=self.device)
            x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std'])
            generated_sample = self.p_sample_one_step(t_sample, x_full, x_part, batch['mean'], batch['std']) # TensorField
            generated_sample = generated_sample.F.reshape(t_sample.shape[0],-1,3) # Tensor

            # add noise to generated samples
            t = torch.randint(0, self.t_steps, size=(generated_sample.shape[0],)).cuda()
            noise = torch.randn(batch['pcd_full'].shape, device=self.device)
            t_sample = generated_sample + self.q_sample(torch.zeros_like(generated_sample), t, noise)
            x_full = self.points_to_tensor(t_sample, batch['mean'], batch['std'])

            # denoise generated samples with simulator and teacher, respectively
            with torch.no_grad():
                self.auxDiff.eval()
                noise_simulator = self.forward_aux(x_full, x_full.sparse(), x_part, t)

                # There will be errors without this expression, reasons not clear yet
                x_full_tmp = self.points_to_tensor(t_sample, batch['mean'], batch['std']) 

                self.teacher.eval()
                noise_teacher = self.forward_teacher(x_full_tmp, x_full_tmp.sparse(), x_part, t)

            # calculate loss
            num_samples = batch['pcd_full'].shape[1] // 10 
            indices = torch.randperm(generated_sample.shape[1])[:num_samples]
            sampled_data_gt = batch['pcd_full'][:, indices, :]
            sampled_data = generated_sample[:, indices, :]

            curvatures_gt = self.compute_curvature(sampled_data_gt, 180)

            num_keypoints = num_samples // 3  
            keypoints_gt, indices_key = self.select_keypoints(sampled_data_gt, curvatures_gt, num_keypoints)
            keypoints = sampled_data[:,indices_key,:]

            differences_gt = keypoints_gt.unsqueeze(2) - keypoints_gt.unsqueeze(1)
            mse_matrix_gt = torch.norm(differences_gt, dim=-1)
            differences = keypoints.unsqueeze(2) - keypoints.unsqueeze(1)
            mse_matrix = torch.norm(differences, dim=-1)         

            scene_wise_loss = F.mse_loss(batch['pcd_full'], generated_sample)
            point_wise_loss = F.mse_loss(mse_matrix_gt, mse_matrix)
            
            distil_loss = ((noise_teacher - noise_simulator) * (t_sample- generated_sample)).mean()

            generator_loss = distil_loss + self.cfg['train']['scene_wise_weight'] * scene_wise_loss + self.cfg['train']['point_wise_weight'] * point_wise_loss

            # log info on progress bar
            self.log('generator_loss', generator_loss, prog_bar=True)

            return generator_loss

    def validation_step(self, batch:dict, batch_idx):
        # print('validating...')

        if batch_idx != 0:
            return

        skip, output_paths = self.valid_paths(batch['filename'])

        self.generator.eval()
        with torch.no_grad():
            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            # for inference we get the partial pcd and sample the noise around the partial
            x_init = batch['pcd_part'].repeat(1,10,1)
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )

            # x_gen_eval = self.p_sample_one_step_val(x_init, x_full, x_part, x_uncond, batch['mean'], batch['std'])
            x_gen_eval = self.p_sample_one_step_val(x_init, x_full, x_part, x_uncond, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,3))

            for i in range(len(batch['pcd_full'])):
                

                pcd_pred = o3d.geometry.PointCloud()
                # pcd_pred_all = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                # pcd_pred_all.points = o3d.utility.Vector3dVector(c_pred)
                dist_pts = np.sqrt(np.sum((c_pred)**2, axis=-1))
                dist_idx = dist_pts < self.hparams['data']['max_range']
                points = c_pred[dist_idx]
                max_z = x_init[i][...,2].max().item()
                min_z = (x_init[i][...,2].mean() - 2 * x_init[i][...,2].std()).item()
                pcd_pred.points = o3d.utility.Vector3dVector(points[(points[:,2] < max_z) & (points[:,2] > min_z)])
                pcd_pred.paint_uniform_color([1.0, 0.,0.])

                pcd_gt = o3d.geometry.PointCloud()
                # pcd_gt_all = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                # pcd_gt_all.points = o3d.utility.Vector3dVector(g_pred)
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
                pcd_gt.paint_uniform_color([0., 1.,0.])

                pcd_part = o3d.geometry.PointCloud()
                pcd_part.points = o3d.utility.Vector3dVector(batch['pcd_part'][i].cpu().detach().numpy())
                pcd_part.paint_uniform_color([0., 1.,0.])

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()

        self.log('val/cd_mean', cd_mean, on_step=True, prog_bar=True)
        self.log('val/cd_std', cd_std, on_step=True)
        self.log('val/precision', pr, on_step=True)
        self.log('val/recall', re, on_step=True)
        self.log('val/fscore', f1, on_step=True)
        torch.cuda.empty_cache()

        return {'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/precision': pr, 'val/recall': re, 'val/fscore': f1}
    
    def valid_paths(self, filenames):
        output_paths = []
        skip = []

        for fname in filenames:
            seq_dir =  f'{self.logger.log_dir}/generated_pcd/{fname.split("/")[-3]}'
            ply_name = f'{fname.split("/")[-1].split(".")[0]}.ply'

            skip.append(path.isfile(f'{seq_dir}/{ply_name}'))
            makedirs(seq_dir, exist_ok=True)
            output_paths.append(f'{seq_dir}/{ply_name}')

        return np.all(skip), output_paths

    def test_step(self, batch:dict, batch_idx):
        self.generator.eval()
        with torch.no_grad():
            skip, output_paths = self.valid_paths(batch['filename'])

            if skip:
                print(f'Skipping generation from {output_paths[0]} to {output_paths[-1]}') 
                return {'test/cd_mean': 0., 'test/cd_std': 0., 'test/precision': 0., 'test/recall': 0., 'test/fscore': 0.}

            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            x_init = batch['pcd_part'].repeat(1,10,1)
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )

            x_gen_eval = self.p_sample_loop(x_init, x_full, x_part, x_uncond, gt_pts, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,3))

            for i in range(len(batch['pcd_full'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                dist_pts = np.sqrt(np.sum((c_pred)**2, axis=-1))
                dist_idx = dist_pts < self.hparams['data']['max_range']
                points = c_pred[dist_idx]
                max_z = x_init[i][...,2].max().item()
                min_z = (x_init[i][...,2].mean() - 2 * x_init[i][...,2].std()).item()
                pcd_pred.points = o3d.utility.Vector3dVector(points[(points[:,2] < max_z) & (points[:,2] > min_z)])
                pcd_pred.paint_uniform_color([1.0, 0.,0.])

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
                pcd_gt.paint_uniform_color([0., 1.,0.])
                
                print(f'Saving {output_paths[i]}')
                o3d.io.write_point_cloud(f'{output_paths[i]}', pcd_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/precision', pr, on_step=True)
        self.log('test/recall', re, on_step=True)
        self.log('test/fscore', f1, on_step=True)
        torch.cuda.empty_cache()

        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/precision': pr, 'test/recall': re, 'test/fscore': f1}

    def configure_optimizers(self):
        optimizer_g = torch.optim.SGD(self.generator.parameters(), lr=self.cfg['train']['lr'])
        optimizer_s = torch.optim.SGD(self.auxDiff.parameters(), lr=self.cfg['train']['lr'])

        return [optimizer_s, optimizer_g]
