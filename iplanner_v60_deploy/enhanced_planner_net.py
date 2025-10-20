# ======================================================================
# Enhanced PlannerNet with Action Prediction
# Based on original PlannerNet by ETH Zurich Robot System Lab
# ======================================================================

import torch
import torch.nn as nn
from percept_net import PerceptNet


class EnhancedPlannerNet(nn.Module):
    def __init__(self, encoder_channel=64, k=5, num_action_classes=7):
        super().__init__()
        self.encoder = PerceptNet(layers=[2, 2, 2, 2])  # depth image encoder 
        self.decoder = EnhancedDecoder(512, encoder_channel, k, num_action_classes)

    def forward(self, x, goal):
        x = self.encoder(x)
        waypoints, collision_prob, actions = self.decoder(x, goal)
        return waypoints, collision_prob, actions


class EnhancedDecoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5, num_action_classes=7):
        super().__init__()
        self.k = k
        self.num_action_classes = num_action_classes
        self.relu = nn.ReLU(inplace=True)
        self.fg = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d((in_channels + goal_channels), 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)

        # Original layers - keep exact same structure
        self.fc1 = nn.Linear(256 * 128, 1024) 
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, k * 2)  # waypoints (2D: x,y only)
        
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)     # collision probability
        
        self.fc_actions1 = nn.Linear(512, 256)
        self.fc_actions2 = nn.Linear(256, k * num_action_classes)

    def forward(self, x, goal):
        # compute goal encoding (exactly like original)
        goal_encoded = self.fg(goal[:, 0:3])
        goal_encoded = goal_encoded[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        
        # cat x with goal in channel dim
        x = torch.cat((x, goal_encoded), dim=1)
        
        # compute features (exactly like original)
        x = self.relu(self.conv1(x))   # size = (N, 512, x.H/32, x.W/32)
        x = self.relu(self.conv2(x))   # size = (N, 512, x.H/60, x.W/60)
        x = torch.flatten(x, 1)

        f = self.relu(self.fc1(x))

        # Waypoints prediction (original path)
        waypoints_features = self.relu(self.fc2(f))
        waypoints = self.fc3(waypoints_features)
        waypoints = waypoints.reshape(-1, self.k, 2)

        # Collision probability prediction (original path)
        c = self.relu(self.frc1(f))
        collision_prob = self.sigmoid(self.frc2(c))

        actions = self.relu(self.fc_actions1(waypoints_features))
        actions = self.fc_actions2(actions)
        actions = actions.reshape(-1, self.k, self.num_action_classes)

        return waypoints, collision_prob, actions


def load_and_enhance_model(model_path, device, num_action_classes=7):
    """
    Load existing PlannerNet model and enhance it with action prediction capability
    """
    # Load the original model
    original_net, best_loss = torch.load(model_path, map_location=device, weights_only=False)
    
    original_fg_weight = original_net.decoder.fg.weight
    actual_encoder_channel = original_fg_weight.shape[0]  # This should be 16, not 64
    
    print(f"Detected encoder_channel: {actual_encoder_channel}")
    
    # Create enhanced model with correct dimensions
    enhanced_net = EnhancedPlannerNet(encoder_channel=actual_encoder_channel, k=5, num_action_classes=num_action_classes)
    
    # Copy weights from original model
    enhanced_state_dict = enhanced_net.state_dict()
    original_state_dict = original_net.state_dict()
    
    for key in original_state_dict:
        if key in enhanced_state_dict:
            if enhanced_state_dict[key].shape == original_state_dict[key].shape:
                enhanced_state_dict[key] = original_state_dict[key]
                print(f"Copied: {key}")
            else:
                print(f"Shape mismatch for {key}: {enhanced_state_dict[key].shape} vs {original_state_dict[key].shape}")
    
    enhanced_net.load_state_dict(enhanced_state_dict)
    
    # Initialize new action prediction layers
    nn.init.xavier_uniform_(enhanced_net.decoder.fc_actions1.weight)
    nn.init.zeros_(enhanced_net.decoder.fc_actions1.bias)
    nn.init.xavier_uniform_(enhanced_net.decoder.fc_actions2.weight)
    nn.init.zeros_(enhanced_net.decoder.fc_actions2.bias)
    
    print("Successfully enhanced model with action prediction capability")
    
    return enhanced_net, best_loss

