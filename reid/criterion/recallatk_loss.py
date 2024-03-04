import warnings
warnings.filterwarnings("ignore")
import torch

def sigmoid(tensor, temp=1.0):
    exponent = -tensor / temp
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

class RecallatK(torch.nn.Module):
    """Recall at K.
    https://github.com/yash0307/RecallatK_surrogate

    Args:
        batch_size (int): Mini-Batchsize to use.
        samples_per_class (int): Number of samples in one class drawn before choosing the next class.
        sigmoid_temperature (float): RS@k: the temperature of the sigmoid used to estimate ranks.
        k_vals (int): Training recall@k vals.
        k_temperatures (int): Temperature for training recall@k vals.
    """
    def __init__(self, batch_size, samples_per_class, sigmoid_temperature=0.01, 
                 k_vals=[1,2,4,8,16], k_temperatures=[1,2,4,8,16]):
        super(RecallatK, self).__init__()
        assert(batch_size%samples_per_class==0)
        self.num_id = int(batch_size/samples_per_class)
        self.sigmoid_temperature = sigmoid_temperature
        self.k_vals = [min(batch_size, k) for k in k_vals]
        self.k_temperatures = k_temperatures

    def forward(self, preds, q_idx):
        loss = 0
        batch_size = preds.shape[0]
        for q_id in range(batch_size):
            samples_per_class = int(batch_size/self.num_id)
            norm_vals = torch.Tensor([min(k, (samples_per_class-1)) for k in self.k_vals]).cuda()
            group_num = int(q_id/samples_per_class)
            # q_id_ = group_num*samples_per_class

            sim_all = (preds[q_id]*preds).sum(1)
            sim_all_g = sim_all.view(self.num_id, int(batch_size/self.num_id))
            sim_diff_all = sim_all.unsqueeze(-1) - sim_all_g[group_num, :].unsqueeze(0).repeat(batch_size,1)
            sim_sg = sigmoid(sim_diff_all, temp=self.sigmoid_temperature)
            for i in range(samples_per_class): sim_sg[group_num*samples_per_class+i,i] = 0.
            sim_all_rk = (1.0 + torch.sum(sim_sg, dim=0)).unsqueeze(dim=0)

            sim_all_rk[:, q_id%samples_per_class] = 0.
            sim_all_rk = sim_all_rk.unsqueeze(dim=-1).repeat(1,1,len(self.k_vals))
            k_vals = torch.Tensor(self.k_vals).cuda()
            k_vals = k_vals.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1, samples_per_class, 1)
            sim_all_rk = k_vals - sim_all_rk
            for given_k in range(0, len(self.k_vals)):
                sim_all_rk[:,:,given_k] = sigmoid(sim_all_rk[:,:,given_k], temp=float(self.k_temperatures[given_k]))

            sim_all_rk[:,q_id%samples_per_class,:] = 0.
            k_vals_loss = torch.Tensor(self.k_vals).cuda()
            k_vals_loss = k_vals_loss.unsqueeze(dim=0)
            recall = torch.sum(sim_all_rk, dim=1)
            recall = torch.minimum(recall, k_vals_loss)
            recall = torch.sum(recall, dim=0)
            recall = torch.div(recall, norm_vals)
            recall = torch.sum(recall)/len(self.k_vals)
            loss += (1.-recall)/batch_size
        return loss