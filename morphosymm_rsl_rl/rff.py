import torch
import torch.nn as nn
import escnn
from escnn import nn as enn
from escnn import group
from escnn.nn import FieldType, EquivariantModule, GeometricTensor
from hydra import compose, initialize

class RandomFourierFeatures(nn.Module):
    def __init__(self, in_features, m=129, sigma=1.0, kernel_type='gaussian'):
        super(RandomFourierFeatures, self).__init__()

        # Validate inputs
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError("in_features must be a positive integer.")
        if not isinstance(m, int) or m <= 0:
            raise ValueError("m must be a positive integer.")
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError("sigma must be a positive number.")
        if kernel_type not in ['gaussian', 'laplacian']:
            raise ValueError("kernel_type must be either 'gaussian' or 'laplacian'.")

        self.in_features = in_features
        self.m = m
        self.sigma = sigma
        self.kernel_type = kernel_type

        if kernel_type == 'gaussian':
            # For Gaussian kernel, sample w from N(0, 1/sigma^2)
            self.register_buffer('w', torch.randn(in_features, m) / sigma)
        elif kernel_type == 'laplacian':
            # For Laplacian kernel, sample w from Cauchy(0, 1/sigma)
            # torch.distributions.Cauchy is used for sampling
            cauchy = torch.distributions.Cauchy(loc=0.0, scale=1.0 / sigma)
            self.register_buffer('w', cauchy.sample((in_features, m)))

        # Sample biases b from a uniform distribution [0, 2*pi]
        self.register_buffer('b', torch.rand(m) * 2 * torch.pi)

    def forward(self, x):
        """
        Computes the random Fourier features for a given input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Lifted features of shape (batch_size, m).
        """
        if x.dim() != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"Input tensor must have shape (batch_size, {self.in_features}), but got {x.shape}")

        # Compute w^T * x + b
        projection = torch.matmul(x, self.w) + self.b

        # Apply the cosine function and scaling factor
        phi_x = torch.sqrt(torch.tensor(2.0 / self.m, device=x.device)) * torch.cos(projection)

        return phi_x

# escnn way
class EquivariantRandomFourierFeatures(nn.Module):
    def __init__(self, task: str, in_features: int, in_type: enn.FieldType, m: int, sigma: float = 1.0, kernel_type: str = 'gaussian'):
        super().__init__()

        # The 'in_type' passed from ActorCriticSymm already contains the perfect
        # gspace, group, and representation mapping for your specific IsaacLab task.
        self.in_type = in_type
        gspace = in_type.gspace
        self.group = gspace.fibergroup

        self.in_features = in_features
        self.m = m # 'm' here is the number of regular representations passed from PPO
        self.sigma = sigma
        self.kernel_type = kernel_type

        # The actual output feature dimension will be m * group.order()
        self.out_type = enn.FieldType(gspace, [self.group.regular_representation] * self.m)
        self.feature_dim = self.out_type.size

        # 1. Equivariant random projection
        self.linear = enn.Linear(in_type, self.out_type, bias=False)

        # Initialize weights according to the specified kernel
        if kernel_type == 'gaussian':
            nn.init.normal_(self.linear.weights, mean=0.0, std=1.0/sigma)
        elif kernel_type == 'laplacian':
            cauchy = torch.distributions.Cauchy(loc=0.0, scale=1.0 / sigma)
            with torch.no_grad():
                self.linear.weights.copy_(cauchy.sample(self.linear.weights.shape))

        # Freeze weights
        for param in self.linear.parameters():
            param.requires_grad = False

        # 2. Equivariant Bias: Sample m uniform biases and repeat them for each group element
        b_base = torch.rand(self.m) * 2 * torch.pi
        b_equiv = b_base.repeat_interleave(self.group.order())
        self.bias = nn.Parameter(b_equiv, requires_grad=False)

    def forward(self, x):
        """
        Computes equivariant RFF. Transparently handles both raw Tensors and GeometricTensors.
        """
        # 1. Check if input is a raw PyTorch tensor and wrap it if necessary
        is_raw_tensor = isinstance(x, torch.Tensor)
        if is_raw_tensor:
            x_geom = escnn.nn.GeometricTensor(x, self.in_type)
        else:
            x_geom = x

        # 2. Compute Equivariant Projection: Wx + b
        projection = self.linear(x_geom).tensor + self.bias

        # Apply the cosine function and scaling factor
        phi_x = torch.sqrt(torch.tensor(2.0 / self.feature_dim, device=x_geom.tensor.device)) * torch.cos(projection)

        # Wrap output in a GeometricTensor
        out_geom = escnn.nn.GeometricTensor(phi_x, self.out_type)

        # 3. If a raw tensor was passed in, return a raw tensor. Otherwise return GeometricTensor.
        if is_raw_tensor:
            return out_geom.tensor
        else:
            return out_geom

# adapts eq. (13) from https://github.com/evelyd/koopman_symmloco/issues/6#issuecomment-3854110341
class AveragedRFF(nn.Module):
    def __init__(self, group_obj: group.Group, in_repr: group.Representation, out_repr: group.Representation, D: int):
        super().__init__()
        self.group = group_obj
        self.in_repr = in_repr
        self.out_repr = out_repr

        # Standard PyTorch linear layer for \omega
        self.linear = nn.Linear(in_repr.size, out_repr.size)
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a standard torch tensor here, shape (batch, in_features)
        batch_size = x.shape[0]
        device = x.device

        z_tilde = torch.zeros(batch_size, self.out_repr.size, device=device)
        T = len(self.group.elements)

        for g in self.group.elements:
            # 1. Get representation matrices for g and its inverse
            # We use g^{-1} on the input to properly align the feature map
            rho_in_inv = torch.tensor(self.in_repr(g.inverse), device=device, dtype=torch.float32)
            rho_out = torch.tensor(self.out_repr(g), device=device, dtype=torch.float32)

            # 2. Transform input: g_i^{-1} x
            g_x = x @ rho_in_inv.T

            # 3. Compute standard RFF: \hat{z}(g_x)
            z_hat = torch.sqrt(torch.tensor(2.0 / self.out_repr.size)) * torch.cos(self.linear(g_x))

            # 4. Transform output and accumulate: g_i \hat{z}
            z_tilde += z_hat @ rho_out.T

        # Divide by T (group size) to complete the average
        return z_tilde / T

class KoopmanEstimator(nn.Module):
    """
    Estimates the Koopman operator K using EDMD, relying on a separate
    lifting function (e.g., RandomFourierFeatures) to compute phi(x).
    """
    def __init__(self, lifting_function: nn.Module, latent_normalizer: nn.Module, action_dim: int, gamma=1.0, device="cuda"):
        """
        Args:
            lifting_function (nn.Module): The function used to compute phi(x).
                                          Must have a 'm' attribute for feature dimension.
            action_dim (int): The dimension of the action vector u.
            N (int): Size of the dataset for regularization scaling.
        """
        super().__init__()

        # Store the lifting function (e.g., an instance of RandomFourierFeatures)
        self.lifting_function = lifting_function
        self.latent_normalizer = latent_normalizer
        self.device = device

        # Get dimensions from the lifting function
        self.state_dim = lifting_function.in_features
        self.feature_dim = lifting_function.m # 'm' from RFF is the dimension of phi(x)
        self.action_dim = action_dim

        # K input: z = [phi(x), u]
        self.koopman_input_dim = self.feature_dim + self.action_dim
        # K output: z+ = phi(x+)
        self.koopman_output_dim = self.feature_dim
        self.gamma = gamma

        # Initialize K as Identity matrix
        A_init = torch.eye(self.koopman_output_dim, device=self.device)
        B_init = torch.zeros(self.koopman_output_dim, self.action_dim, device=self.device)
        self.K_matrix = torch.cat([A_init, B_init], dim=1) # Shape: (feature_dim, feature_dim + action_dim)
        self.Gbar_x = torch.zeros(self.koopman_input_dim, self.koopman_input_dim, device=self.device)
        self.Gbar_xy = torch.zeros(self.koopman_output_dim, self.koopman_input_dim, device=self.device)

    def compute_koopman_op(self, states, actions, next_states):
        """
        Computes the Koopman operator K using regularized least squares (EDMD).

        Args:
            states (torch.Tensor): Normed states x at time t, shape (num_samples, state_dim).
            actions (torch.Tensor): Normed actions u at time t, shape (num_samples, action_dim).
            next_states (torch.Tensor): Normed states x+ at time t+1, shape (num_samples, state_dim).

        Returns:
            torch.Tensor: Koopman operator matrix K of shape
                          (koopman_output_dim, koopman_input_dim).
        """

        # --- Input Validation (Refined to use class attributes) ---
        if states.dim() != 2 or next_states.dim() != 2:
            raise ValueError("states and next_states must be 2D tensors.")
        if states.shape != next_states.shape:
            raise ValueError("states and next_states must have the same shape.")
        if states.shape[1] != self.state_dim:
            raise ValueError(f"states must have {self.state_dim} features in the second dimension.")
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"actions must have {self.action_dim} features in the second dimension.")

        # --- Data Lifting (Delegated to the RFF instance) ---
        # Compute the lifted representations phi(x) and phi(x+)
        lifted_states = self.lifting_function(states)     # phi(x), shape (N, feature_dim)
        lifted_next_states = self.lifting_function(next_states) # phi(x+), shape (N, feature_dim)

        # Normalize the lifted states TODO try without latent normalization
        # lifted_states = self.latent_normalizer(lifted_states)
        # lifted_next_states = self.latent_normalizer(lifted_next_states)

        # Koopman input vector z = [phi(x), u]
        lifted_data = torch.cat([lifted_states, actions], dim=1) # Phi_x.T, shape (N, input_dim)

        # Koopman target vector z+ = phi(x+)
        next_lifted_data = lifted_next_states # Phi_y.T, shape (N, output_dim)

        # --- Matrix Construction (EDMD) ---
        # Phi_x is the matrix of current *input* vectors z (columns)
        Phi_x = lifted_data.T  # shape: (input_dim, num_samples)

        # Phi_y is the matrix of next *target* vectors z+ (columns)
        Phi_y = next_lifted_data.T  # shape: (output_dim, num_samples)

        # compute Koopman operator with Gramians
        G_x = (Phi_x @ Phi_x.T)
        G_xy = (Phi_y @ Phi_x.T)
        self.Gbar_x = self.Gbar_x+ G_x
        self.Gbar_xy = self.Gbar_xy + G_xy

        # --- Koopman Operator Computation ---
        # Regularization term
        self.reg = self.gamma * torch.eye(Phi_x.shape[0], device=self.device)
        K = self.Gbar_xy @ torch.linalg.pinv(self.Gbar_x + self.reg, rcond=0.95, hermitian=True)

        # Store the computed operator and return
        self.K_matrix = K
        return K

    def predict_next_lifted_state(self, state, action):
        if self.K_matrix is None:
            raise RuntimeError("Koopman operator K must be computed first.")

        # 1. Compute the lifted state (returns GeometricTensor for ERFF)
        lifted_state = self.lifting_function(state)

        # 2. EXTRACT THE RAW TENSORS (This fixes the TypeError!)
        lifted_tensor = lifted_state.tensor if hasattr(lifted_state, 'tensor') else lifted_state
        action_tensor = action.tensor if hasattr(action, 'tensor') else action

        # 3. Concatenate the raw tensors
        z = torch.cat([lifted_tensor, action_tensor], dim=-1)

        # 4. Multiply by K matrix
        z_pred = self.K_matrix @ z.T

        # Return as a raw tensor (transposed back to N x out_dim)
        return z_pred.T

    def predict_from_lifted_state(self, lifted_state, action):
        """Predicts the next latent state using an already computed lifted state."""
        if self.K_matrix is None:
            raise RuntimeError("Koopman operator K must be computed first.")

        # Get the raw tensor from the lifted state (handles both GeometricTensor and raw tensor cases)
        lifted_tensor = lifted_state.tensor if hasattr(lifted_state, 'tensor') else lifted_state
        action_tensor = action.tensor if hasattr(action, 'tensor') else action
        z = torch.cat([lifted_tensor, action_tensor], dim=-1)

        # Compute the Koopman prediction
        z_pred = self.K_matrix @ z.T

        return z_pred.T

    def compute_pred_error(self, states, actions, next_states):
        """
        Computes the mean prediction error ||K z - z+||^2 over a dataset.

        Args:
            states (torch.Tensor): States x at time t, shape (num_samples, state_dim).
            actions (torch.Tensor): Actions u at time t, shape (num_samples, action_dim).
            next_states (torch.Tensor): States x+ at time t+1, shape (num_samples, state_dim).

        Returns:
            torch.Tensor: The mean prediction error (scalar).
        """
        # Compute the lifted representations
        lifted_states = self.lifting_function(states)
        lifted_next_states = self.lifting_function(next_states)

        # Normalize the lifted states TODO try without norming the latent states
        # lifted_states = self.latent_normalizer(lifted_states)
        # lifted_next_states = self.latent_normalizer(lifted_next_states)

        # Compute the Koopman input vector z = [phi(x), u]
        lifted_data = torch.cat([lifted_states, actions], dim=1)

        # Compute the predicted next lifted state
        z_pred = self.K_matrix @ lifted_data.T

        # Compute the prediction error for each sample
        pred_error = torch.norm(z_pred - lifted_next_states.T, dim=0) ** 2

        # Return the mean prediction error as a scalar
        return pred_error.mean()

class EquivariantKoopmanEstimator(KoopmanEstimator):
    def __init__(self, lifting_function: EquivariantRandomFourierFeatures, latent_normalizer, action_type: enn.FieldType, gamma=1.0, device="cuda"):
        # Initialize standard eDMD using the base class
        super().__init__(lifting_function, latent_normalizer, action_type.size, gamma, device)

        self.action_type = action_type
        self.phi_type = lifting_function.out_type
        self.group = self.phi_type.gspace.fibergroup

        self.feature_dim = self.phi_type.size
        self.koopman_input_dim = self.feature_dim + self.action_type.size
        self.koopman_output_dim = self.feature_dim

        # Initialize K as Identity matrix
        A_init = torch.eye(self.koopman_output_dim, device=self.device)
        B_init = torch.zeros(self.koopman_output_dim, self.action_dim, device=self.device)
        self.K_matrix = torch.cat([A_init, B_init], dim=1) # Shape: (feature_dim, feature_dim + action_dim)
        self.Gbar_x = torch.zeros(self.koopman_input_dim, self.koopman_input_dim, device=self.device)
        self.Gbar_xy = torch.zeros(self.koopman_output_dim, self.koopman_input_dim, device=self.device)

    def compute_koopman_op(self, states_tensor, actions_tensor, next_states_tensor):
        """ Computes the standard eDMD K, then projects it to be strictly equivariant. """

        # 1. Compute standard K using the parent class method
        # (Note: lifting_function handles the wrapping internally if modified slightly,
        # but you should ensure states are
        K_standard = super().compute_koopman_op(states_tensor, actions_tensor, next_states_tensor)

        device = K_standard.device
        K_eq = torch.zeros_like(K_standard)

        # 2. Project K using Group Averaging (Haar Integration)
        for g in self.group.elements:
            # \rho_{out}(g) for the target \phi(x+)
            rho_out = torch.tensor(self.phi_type.representation(g), device=device, dtype=torch.float32)

            # \rho_{\phi}(g^{-1}) and \rho_{u}(g^{-1}) for the input [ \phi(x), u ]
            g_inv = ~g
            rho_phi_inv = torch.tensor(self.phi_type.representation(g_inv), device=device, dtype=torch.float32)
            rho_u_inv = torch.tensor(self.action_type.representation(g_inv), device=device, dtype=torch.float32)

            # Combine input representations into a block diagonal matrix
            rho_in_inv = torch.block_diag(rho_phi_inv, rho_u_inv)

            # Accumulate the projected matrix
            K_eq += rho_out @ K_standard @ rho_in_inv

        # Average over the group
        K_eq = K_eq / self.group.order()

        self.K_matrix = K_eq
        return K_eq

class RunningLatentNormalizer(nn.Module):
    def __init__(self, num_features, device='cpu'):
        super().__init__()
        self.num_features = num_features
        self.device = device

        # Register buffers to store the running mean, variance, and count
        # These will be saved with the model state dictionary
        self.register_buffer('mean', torch.zeros(num_features, device=self.device))
        self.register_buffer('var', torch.ones(num_features, device=self.device))
        self.register_buffer('count', torch.tensor(0.0, device=self.device))
        self.epsilon = 1e-8 # A small value to prevent division by zero

    def update(self, x):
        """
        Updates the running mean and variance with a new batch of data.

        Args:
            x (torch.Tensor): A batch of input data, shape (batch_size, num_features).
        """
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Welford's algorithm for numerically stable updates
        new_mean = self.mean + delta * batch_count / total_count

        # The variance update is a bit more complex, using the "Welford's algorithm"
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        # Update the running statistics
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count.clone().detach())

    def forward(self, x):
        """
        Normalizes the input tensor using the current running mean and variance.

        Args:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        # We use sqrt(self.var + self.epsilon) to prevent division by zero
        normalized_x = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return normalized_x

    def to_device(self, device):
        """Moves the normalizer's buffers to the specified device."""
        self.to(device)
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.count = self.count.to(device)
        self.device = device

# Example Usage:
if __name__ == '__main__':

    # Define hyperparameters
    in_features = 47
    m = 129
    sigma = 1.0

    # Instantiate the RFF class
    rff_layer = RandomFourierFeatures(in_features, m, sigma)

    # Create a dummy input tensor
    batch_size = 100
    dummy_input = torch.randn(batch_size, in_features)

    # Compute the random Fourier features
    lifted_features = rff_layer(dummy_input)

    # Print the shape of the output
    print(f"Shape of the input: {dummy_input.shape}")
    print(f"Shape of the lifted features: {lifted_features.shape}")

    # You can now use the lifted_features with a linear model, e.g., a simple linear regression
    # dummy_linear_model = nn.Linear(m, 1)
    # output = dummy_linear_model(lifted_features)
    # print(f"Shape of the final output: {output.shape}")
    num_features = 129 # Corresponds to your state dimension

    # 1. Instantiate the normalizer
    normalizer = RunningLatentNormalizer(num_features)
    print("Initial state:", normalizer.mean, normalizer.var, normalizer.count)

    # 2. Simulate training iterations
    for i in range(5):
        # Generate a new batch of data (simulating a changing distribution)
        # Note: In a real RL setup, this would be your `lifted_features`
        new_data = torch.randn(100, num_features) * (i + 1) # Example of a changing distribution

        # 3. Update the normalizer's stats with the new batch
        normalizer.update(new_data)

        # 4. Normalize the data for the next training step
        normalized_data = normalizer(new_data)

        print(f"\nIteration {i+1}:")
        print("Updated mean:", normalizer.mean.mean().item())
        print("Updated variance:", normalizer.var.mean().item())
        print("Normalized data mean:", normalized_data.mean().item())
        print("Normalized data variance:", normalized_data.var().item())