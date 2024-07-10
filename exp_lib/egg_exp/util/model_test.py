import torch
import torch.nn.functional as F
import os
import numpy as np
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import torch.distributed as dist
import sys
from .ddp_util import all_gather

def calculate_EER(scores, labels):
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100

def df_test(framework, loader, run_on_ddp=False, get_scores=False):
    '''Test deepfake detection and return EER 
    
    Param
        loader: DataLoader that returns (wav, label)
        get_scores: if True, returns the score and label used in EER calculation.
        
    Return
        eer(float)
        score(list(float))
        labels(list(int))
    '''
    framework.eval()

    labels = []
    scores = []
    filenames = []
    speakers = []
    with tqdm(total=len(loader), ncols=90) as pbar, torch.set_grad_enabled(False):
        for x, label, filename, spk_id in loader:
            if run_on_ddp:
                x = x.to(torch.float32).to(framework.device, non_blocking=True)
            else:
                x = x.to(torch.float32).to(framework.device)
            x = framework(x).to('cpu')
            
            for i in range(x.size(0)):
                if x.size(1) == 1:
                    scores.append(x[i, 0].item())
                else:
                    scores.append(x[i, 1].item())
                labels.append(label[i])
                filenames.append(filename[i])
                speakers.append(spk_id[i])
            pbar.update(1)
    
    if run_on_ddp:
        _synchronize()
        scores = all_gather(scores)
        labels = all_gather(labels)
        filenames = all_gather(filenames)
        speakers = all_gather(speakers)

    with open('/code/temp/scores.txt', "w") as fh:
        for fn, sco, spk_id, key in zip(filenames, scores, speakers, labels):
            fh.write("{} {} {} {}\n".format(spk_id, fn, sco, key))

    dcf, eer, cllr = calculate_minDCF_EER_CLLR(cm_scores_file='/code/temp/scores.txt', output_file="whatever")

    eer_repo = calculate_EER(scores, [0 if x == "spoof" else 1 for x in labels])

    if get_scores:
        return eer_repo, scores, labels, filenames, eer, dcf, cllr
    else:
        return eer
    
def df_test_embd(framework, loader, run_on_ddp=False, get_scores=False):
    '''Test deepfake detection and return EER 
    
    Param
        loader: DataLoader that returns (wav, label)
        get_scores: if True, returns the score and label used in EER calculation.
        
    Return
        eer(float)
        score(list(float))
        labels(list(int))
    '''
    framework.eval()

    labels = [[],[],[],[],[]]
    scores = [[],[],[],[],[]]
    with tqdm(total=len(loader), ncols=90) as pbar, torch.set_grad_enabled(False):
        for x, label in loader:
            if run_on_ddp:
                x = x.to(torch.float32).to(framework.device, non_blocking=True)
            else:
                x = x.to(torch.float32).to(framework.device)
            xs = framework(x, all_loss=True)
            for j in range(5):
                x = xs[j].to('cpu')

                for i in range(x.size(0)):
                    if x.size(1) == 1:
                        scores[j].append(x[i, 0].item())
                    else:
                        scores[j].append(x[i, 1].item())
                    labels[j].append(label[i].item())
            
            pbar.update(1)
    
    if run_on_ddp:
        _synchronize()
        for j in range(5):
            scores[j] = all_gather(scores[j])
            labels[j] = all_gather(labels[j])
        print('s0',len(scores[0]),'    s1',len(scores[1]))

    EER = []
    for j in range(5):
        eer = calculate_EER(scores[j], labels[j])
        EER.append(eer)
        
    if get_scores:
        return EER, scores, labels
    else:
        return EER

def sv_enrollment(framework, loader):
    '''
    '''
    framework.eval()
    
    keys = []
    embeddings_full = []
    embeddings_seg = []

    with tqdm(total=len(loader), ncols=90) as pbar, torch.set_grad_enabled(False):
        for x_full, x_seg, key in loader:
            x_full = x_full.to(torch.float32).to(framework.device, non_blocking=True)
            x_seg = x_seg.to(torch.float32).to(framework.device, non_blocking=True).view(-1, x_seg.size(-1)) 
            
            x_full = framework(x_full).to('cpu')
            x_seg = framework(x_seg).to('cpu')
            
            keys.append(key[0])
            embeddings_full.append(x_full)
            embeddings_seg.append(x_seg)

            pbar.update(1)

    _synchronize()

    # gather
    keys = all_gather(keys)
    embeddings_full = all_gather(embeddings_full)
    embeddings_seg = all_gather(embeddings_seg)

    full_dict = {}
    seg_dict = {}
    for i in range(len(keys)):
        full_dict[keys[i]] = embeddings_full[i]
        seg_dict[keys[i]] = embeddings_seg[i]
            
    return full_dict, seg_dict
    
def sv_test(trials, single_embedding=None, multi_embedding=None):
    '''Calculate EER for test speaker verification performance.
    
    Param
        trials(list): list of SV_Trial (it contains key1, key2, label) 
        single_embedding(dict): embedding dict extracted from single utterance
        multi_embedding(dict): embedding dict extracted from multi utterance, such as TTA
    
    Return
        eer(float)
    '''
    labels = []
    cos_sims_full = [[], []]
    cos_sims_seg = [[], []]

    for item in trials:
        if single_embedding is not None:
            cos_sims_full[0].append(single_embedding[item.key1])
            cos_sims_full[1].append(single_embedding[item.key2])

        if multi_embedding is not None:
            cos_sims_seg[0].append(multi_embedding[item.key1])
            cos_sims_seg[1].append(multi_embedding[item.key2])

        labels.append(item.label)

    # cosine_similarity - full
    count = 0
    cos_sims = 0
    if single_embedding is not None:
        buffer1 = torch.cat(cos_sims_full[0], dim=0)
        buffer2 = torch.cat(cos_sims_full[1], dim=0)
        cos_sims_full = F.cosine_similarity(buffer1, buffer2)
        cos_sims = cos_sims + cos_sims_full
        count += 1

    # cosine_similarity - seg
    if multi_embedding is not None:
        batch = len(labels)
        num_seg = cos_sims_seg[0][0].size(0)
        buffer1 = torch.stack(cos_sims_seg[0], dim=0).view(batch, num_seg, -1)
        buffer2 = torch.stack(cos_sims_seg[1], dim=0).view(batch, num_seg, -1)
        buffer1 = buffer1.repeat(1, num_seg, 1).view(batch * num_seg * num_seg, -1)
        buffer2 = buffer2.repeat(1, 1, num_seg).view(batch * num_seg * num_seg, -1)
        cos_sims_seg = F.cosine_similarity(buffer1, buffer2)
        cos_sims_seg = cos_sims_seg.view(batch, num_seg * num_seg)
        cos_sims_seg = cos_sims_seg.mean(dim=1)
        cos_sims = cos_sims + cos_sims_seg
        count += 1

    cos_sims = (cos_sims_full + cos_sims_seg) / count
    eer = calculate_EER(cos_sims, labels)
    
    return eer

def _synchronize():
    torch.cuda.empty_cache()
    dist.barrier()

def calculate_minDCF_EER_CLLR(cm_scores_file,
                       output_file,
                       printout=False):
    # Evaluation metrics for Phase 1
    # Primary metrics: min DCF,
    # Secondary metrics: EER, CLLR

    Pspoof = 0.05
    dcf_cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Cmiss': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa' : 10, # Cost of CM system falsely accepting nontarget speaker
    }


    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_keys = cm_data[:, 3]
    cm_scores = cm_data[:, 2].astype(np.float64)

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_cm, frr, far, thresholds = compute_eer(bona_cm, spoof_cm)#[0]
    cllr_cm = calculate_CLLR(bona_cm, spoof_cm)
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tmin DCF \t\t= {} % '
                        '(min DCF for countermeasure)\n'.format(
                            minDCF_cm))
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(EER for countermeasure)\n'.format(
                            eer_cm * 100))
            f_res.write('\tCLLR\t\t= {:8.9f} % '
                        '(CLLR for countermeasure)\n'.format(
                            cllr_cm * 100))
        os.system(f"cat {output_file}")

    return minDCF_cm, eer_cm, cllr_cm

def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_Pmiss_Pfa_Pspoof_curves(tar_scores, non_scores, spf_scores):

    # Concatenate all scores and designate arbitrary labels 1=target, 0=nontarget, -1=spoof
    all_scores = np.concatenate((tar_scores, non_scores, spf_scores))
    labels = np.concatenate((np.ones(tar_scores.size), np.zeros(non_scores.size), -1*np.ones(spf_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Cumulative sums
    tar_sums    = np.cumsum(labels==1)
    non_sums    = np.cumsum(labels==0)
    spoof_sums  = np.cumsum(labels==-1)

    Pmiss       = np.concatenate((np.atleast_1d(0), tar_sums / tar_scores.size))
    Pfa_non     = np.concatenate((np.atleast_1d(1), 1 - (non_sums / non_scores.size)))
    Pfa_spoof   = np.concatenate((np.atleast_1d(1), 1 - (spoof_sums / spf_scores.size)))
    thresholds  = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return Pmiss, Pfa_non, Pfa_spoof, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, frr, far, thresholds


def compute_mindcf(frr, far, thresholds, Pspoof, Cmiss, Cfa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds

    p_target = 1- Pspoof
    for i in range(0, len(frr)):
        # Weighted sum of false negative and false positive errors.
        c_det = Cmiss * frr[i] * p_target + Cfa * far[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(Cmiss * p_target, Cfa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                 Pmiss_spoof_asv, cost_model, print_cost):

    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit(
            'ERROR: Your prior probabilities should be positive and sum up to one.'
        )

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit(
            'ERROR: you should provide miss rate of spoof tests against your ASV system.'
        )

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit(
            'ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
        cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit(
            'You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?'
        )

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(
            bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.
              format(cost_model['Ptar']))
        print(
            '   Pnon         = {:8.5f} (Prior probability of nontarget user)'.
            format(cost_model['Pnon']))
        print(
            '   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.
            format(cost_model['Pspoof']))
        print(
            '   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'
            .format(cost_model['Cfa_asv']))
        print(
            '   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'
            .format(cost_model['Cmiss_asv']))
        print(
            '   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'
            .format(cost_model['Cfa_cm']))
        print(
            '   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'
            .format(cost_model['Cmiss_cm']))
        print(
            '\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)'
        )

        if C2 == np.minimum(C1, C2):
            print(
                '   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(
                    C1 / C2))
        else:
            print(
                '   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(
                    C2 / C1))

    return tDCF_norm, CM_thresholds


def calculate_CLLR(target_llrs, nontarget_llrs):
    """
    Calculate the CLLR of the scores.
    
    Parameters:
    target_llrs (list or numpy array): Log-likelihood ratios for target trials.
    nontarget_llrs (list or numpy array): Log-likelihood ratios for non-target trials.
    
    Returns:
    float: The calculated CLLR value.
    """
    def negative_log_sigmoid(lodds):
        """
        Calculate the negative log of the sigmoid function.
        
        Parameters:
        lodds (numpy array): Log-odds values.
        
        Returns:
        numpy array: The negative log of the sigmoid values.
        """
        return np.log1p(np.exp(-lodds))

    # Convert the input lists to numpy arrays if they are not already
    target_llrs = np.array(target_llrs)
    nontarget_llrs = np.array(nontarget_llrs)
    
    # Calculate the CLLR value
    cllr = 0.5 * (np.mean(negative_log_sigmoid(target_llrs)) + np.mean(negative_log_sigmoid(-nontarget_llrs))) / np.log(2)
    
    return cllr


def compute_Pmiss_Pfa_Pspoof_curves(tar_scores, non_scores, spf_scores):

    # Concatenate all scores and designate arbitrary labels 1=target, 0=nontarget, -1=spoof
    all_scores = np.concatenate((tar_scores, non_scores, spf_scores))
    labels = np.concatenate((np.ones(tar_scores.size), np.zeros(non_scores.size), -1*np.ones(spf_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Cumulative sums
    tar_sums    = np.cumsum(labels==1)
    non_sums    = np.cumsum(labels==0)
    spoof_sums  = np.cumsum(labels==-1)

    Pmiss       = np.concatenate((np.atleast_1d(0), tar_sums / tar_scores.size))
    Pfa_non     = np.concatenate((np.atleast_1d(1), 1 - (non_sums / non_scores.size)))
    Pfa_spoof   = np.concatenate((np.atleast_1d(1), 1 - (spoof_sums / spf_scores.size)))
    thresholds  = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return Pmiss, Pfa_non, Pfa_spoof, thresholds


def compute_teer(Pmiss_CM, Pfa_CM, tau_CM, Pmiss_ASV, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV):
    # Different spoofing prevalence priors (rho) parameters values
    rho_vals            = [0,0.5,1]

    tEER_val    = np.empty([len(rho_vals),len(tau_ASV)], dtype=float)

    for rho_idx, rho_spf in enumerate(rho_vals):

        # Table to store the CM threshold index, per each of the ASV operating points
        tEER_idx_CM = np.empty(len(tau_ASV), dtype=int)

        tEER_path   = np.empty([len(rho_vals),len(tau_ASV),2], dtype=float)

        # Tables to store the t-EER, total Pfa and total miss valuees along the t-EER path
        Pmiss_total = np.empty(len(tau_ASV), dtype=float)
        Pfa_total   = np.empty(len(tau_ASV), dtype=float)
        min_tEER    = np.inf
        argmin_tEER = np.empty(2)

        # best intersection point
        xpoint_crit_best = np.inf
        xpoint = np.empty(2)

        # Loop over all possible ASV thresholds
        for tau_ASV_idx, tau_ASV_val in enumerate(tau_ASV):

            # Tandem miss and fa rates as defined in the manuscript
            Pmiss_tdm = Pmiss_CM + (1 - Pmiss_CM) * Pmiss_ASV[tau_ASV_idx]
            Pfa_tdm   = (1 - rho_spf) * (1 - Pmiss_CM) * Pfa_non_ASV[tau_ASV_idx] + rho_spf * Pfa_CM * Pfa_spf_ASV[tau_ASV_idx]

            # Store only the INDEX of the CM threshold (for the current ASV threshold)
            h = Pmiss_tdm - Pfa_tdm
            tmp = np.argmin(abs(h))
            tEER_idx_CM[tau_ASV_idx] = tmp

            if Pmiss_ASV[tau_ASV_idx] < (1 - rho_spf) * Pfa_non_ASV[tau_ASV_idx] + rho_spf * Pfa_spf_ASV[tau_ASV_idx]:
                Pmiss_total[tau_ASV_idx] = Pmiss_tdm[tmp]
                Pfa_total[tau_ASV_idx] = Pfa_tdm[tmp]

                tEER_val[rho_idx,tau_ASV_idx] = np.mean([Pfa_total[tau_ASV_idx], Pmiss_total[tau_ASV_idx]])

                tEER_path[rho_idx,tau_ASV_idx, 0] = tau_ASV_val
                tEER_path[rho_idx,tau_ASV_idx, 1] = tau_CM[tmp]

                if tEER_val[rho_idx,tau_ASV_idx] < min_tEER:
                    min_tEER = tEER_val[rho_idx,tau_ASV_idx]
                    argmin_tEER[0] = tau_ASV_val
                    argmin_tEER[1] = tau_CM[tmp]

                # Check how close we are to the INTERSECTION POINT for different prior (rho) values:
                LHS = Pfa_non_ASV[tau_ASV_idx]/Pfa_spf_ASV[tau_ASV_idx]
                RHS = Pfa_CM[tmp]/(1 - Pmiss_CM[tmp])
                crit = abs(LHS - RHS)

                if crit < xpoint_crit_best:
                    xpoint_crit_best = crit
                    xpoint[0] = tau_ASV_val
                    xpoint[1] = tau_CM[tmp]
                    xpoint_tEER = Pfa_spf_ASV[tau_ASV_idx]*Pfa_CM[tmp]
            else:
                # Not in allowed region
                tEER_path[rho_idx,tau_ASV_idx, 0] = np.nan
                tEER_path[rho_idx,tau_ASV_idx, 1] = np.nan
                Pmiss_total[tau_ASV_idx] = np.nan
                Pfa_total[tau_ASV_idx] = np.nan
                tEER_val[rho_idx,tau_ASV_idx] = np.nan

        return xpoint_tEER*100