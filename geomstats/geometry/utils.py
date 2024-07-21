import torch

def set_diag(x, new_diag):
    """Set the diagonal along the last two axis.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    new_diag : array-like, shape=[dim[-2]]
        Values to set on the diagonal.

    Returns
    -------
    None

    Notes
    -----
    This mimics tensorflow.linalg.set_diag(x, new_diag), when new_diag is a
    1-D array, but modifies x instead of creating a copy.
    """
    arr_shape = x.shape
    off_diag = (1 - torch.eye(arr_shape[-1]).to(x.device)) * x
    diag = torch.einsum("ij,...i->...ij", 
                        torch.eye(new_diag.shape[-1]).to(new_diag.device), new_diag)
    return diag + off_diag