import torch
from torch.autograd import Function
import scipy.linalg

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        # m = input.detach().cpu().numpy().astype(np.float_)
        m = input.detach().cpu().numpy()
        # sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).astype(m.dtype)).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            # sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            # gm = grad_output.data.cpu().numpy().astype(np.float_)
            sqrtm = sqrtm.data.cpu().numpy()
            gm = grad_output.data.cpu().numpy()

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


def main():
    from torch.autograd import gradcheck
    k = torch.randn(20, 10).double()
    # Create a positive definite matrix
    pd_mat = (k.t().matmul(k)).requires_grad_()
    test = gradcheck(sqrtm, (pd_mat,))
    print(test)


if __name__ == '__main__':
    main()