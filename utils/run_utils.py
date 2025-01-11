import torch

def make_type_mask_for_pad_sequence(pad_seqs, num_user):
    """ Make the type mask. """
    type_mask = torch.zeros([*pad_seqs.shape, num_user], dtype=torch.int32)
    type_mask = type_mask.to(pad_seqs.device)
    for i in range(num_user):
        type_mask[:, :, i] = pad_seqs == i
    type_mask[:, :, 0] = 0

    return type_mask

def compute_loglikelihood(time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, lambda_type_mask):
    eps = torch.finfo(torch.float32).eps
    event_lambdas = torch.sum(lambda_at_event * lambda_type_mask, dim=-1) + eps
    event_lambdas = event_lambdas.masked_fill_(~seq_mask, 1.0)
    event_ll = torch.log(event_lambdas)
    lambdas_total_samples = lambdas_loss_samples.sum(dim=-1)
    non_event_ll = lambdas_total_samples.mean(dim=-1) * time_delta_seq * seq_mask
    num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]

    return event_ll, non_event_ll, num_events