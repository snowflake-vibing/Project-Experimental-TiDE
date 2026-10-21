"""
Script Giả Lập Chi Tiết Forward Pass Của Mô Hình TiDE (Time-series Dense Encoder)
================================================================================
Mục đích:
  Theo dõi và in tường minh kích thước (shape) của Tensor qua từng biến đổi toán học
  trong Section 4 của bài báo TiDE.

Cấu hình thực nghiệm mô phỏng:
  - Batch size (B): 32
  - Độ dài nhìn lại quá khứ (L): 720
  - Độ dài khoảng dự báo (H): 96
  - Số chiều dynamic covariates gốc (r): 4
  - Số chiều static attributes (r_stat): 2
  - Số chiều dynamic covariates sau khi nén (r_tilde): 4
  - Chiều ẩn của Encoder/Decoder (hidden_size): 256
  - Số chiều đặc trưng giải mã mỗi bước tương lai (decoder_output_dim / p): 16
  - Số tầng Encoder (num_encoder_layers / n_e): 2
  - Số tầng Decoder (num_decoder_layers / n_d): 2
  - Chiều ẩn của Temporal Decoder (temporal_hidden): 64
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch.nn as nn
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """Khối Residual Block đa tầng (MLP) với Skip Connection và Layer Normalization (Section 4.1).
    
    Công thức:
        v = LayerNorm( Dropout(W2 * ReLU(W1 * u + b1) + b2) + W_skip * u )
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1, use_layer_norm: bool = True):
        super().__init__()
        # Nhánh phi tuyến MLP
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Nhánh tắt Skip Connection
        if in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim)
        else:
            self.skip_proj = nn.Identity()
            
        # Chuẩn hóa LayerNorm
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [..., in_dim]
        mlp_out = self.dropout(self.fc2(self.relu(self.fc1(u))))
        skip_out = self.skip_proj(u)
        v_raw = mlp_out + skip_out
        if self.use_layer_norm:
            return self.layer_norm(v_raw)
        return v_raw


class TiDEModelSimulation(nn.Module):
    """Mô hình TiDE hoàn chỉnh bóc tách từng bước Forward Pass kèm in kích thước Tensor."""
    
    def __init__(
        self,
        L: int = 720,
        H: int = 96,
        r: int = 4,
        r_stat: int = 2,
        r_tilde: int = 4,
        hidden_size: int = 256,
        p: int = 16,
        n_e: int = 2,
        n_d: int = 2,
        temporal_hidden: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.L = L
        self.H = H
        self.r = r
        self.r_stat = r_stat
        self.r_tilde = r_tilde
        self.hidden_size = hidden_size
        self.p = p
        
        # 1. Feature Projection (Section 4.2 Step 1): r -> r_tilde
        self.feature_proj = ResidualBlock(
            in_dim=r, 
            hidden_dim=r_tilde * 2, 
            out_dim=r_tilde, 
            dropout=dropout
        )
        
        # 2. Dense Encoder (Section 4.2 Step 2):
        # Input dim = L + (L + H) * r_tilde + r_stat
        encoder_in_dim = L + (L + H) * r_tilde + r_stat
        encoder_layers = []
        for i in range(n_e):
            in_d = encoder_in_dim if i == 0 else hidden_size
            encoder_layers.append(
                ResidualBlock(in_dim=in_d, hidden_dim=hidden_size, out_dim=hidden_size, dropout=dropout)
            )
        self.dense_encoder = nn.Sequential(*encoder_layers)
        
        # 3. Dense Decoder (Section 4.3 Step 1):
        # Output dim = H * p
        decoder_layers = []
        for i in range(n_d - 1):
            decoder_layers.append(
                ResidualBlock(in_dim=hidden_size, hidden_dim=hidden_size, out_dim=hidden_size, dropout=dropout)
            )
        # Khối cuối cùng của decoder đưa về H * p
        decoder_layers.append(
            ResidualBlock(in_dim=hidden_size, hidden_dim=hidden_size, out_dim=H * p, dropout=dropout)
        )
        self.dense_decoder = nn.Sequential(*decoder_layers)
        
        # 4. Temporal Decoder (Section 4.3 Step 2):
        # Input = d_t (p chiều) + x_tilde_(L+t) (r_tilde chiều) -> 1 chiều
        self.temporal_decoder = nn.Sequential(
            ResidualBlock(in_dim=p + r_tilde, hidden_dim=temporal_hidden, out_dim=temporal_hidden, dropout=dropout),
            nn.Linear(temporal_hidden, 1)
        )
        
        # 5. Global Linear Skip Connection (Section 4.4): L -> H
        self.global_skip = nn.Linear(L, H)

    def forward_with_logging(
        self,
        y_past: torch.Tensor,
        x_all: torch.Tensor,
        a_stat: torch.Tensor,
        verbose: bool = True
    ) -> torch.Tensor:
        """Thực thi forward pass và in chi tiết kích thước Tensor qua từng phép toán."""
        
        if verbose:
            print("=" * 85)
            print("BẮT ĐẦU FORWARD PASS CHI TIẾT MÔ HÌNH TiDE")
            print("=" * 85)
            print(f"[*] CẤU HÌNH ĐẦU VÀO BAN ĐẦU:")
            print(f"    - y_past (Chuỗi lịch sử quan sát)           : {tuple(y_past.shape)}  [B, L]")
            print(f"    - x_all  (Biến ngoại sinh động toàn thời gian): {tuple(x_all.shape)}  [B, L+H, r]")
            print(f"    - a_stat (Thuộc tính ngoại sinh tĩnh)        : {tuple(a_stat.shape)}  [B, r_stat]")
            print("-" * 85)

        # -------------------------------------------------------------------------
        # BƯỚC 1: Chuẩn Hóa Thực Thể Đảo Nghịch RevIN (Section 4.5)
        # Công thức: mu = mean(y), sigma = std(y); y_norm = (y - mu) / sigma
        # -------------------------------------------------------------------------
        mu = y_past.mean(dim=1, keepdim=True)              # [B, 1]
        sigma = torch.sqrt(y_past.var(dim=1, keepdim=True, unbiased=False) + 1e-5) # [B, 1]
        y_norm = (y_past - mu) / sigma                      # [B, L]
        
        if verbose:
            print(f"[Bước 1 - Section 4.5: Chuẩn Hóa RevIN]")
            print(f"    Công thức: y_norm = (y_past - mu) / sigma")
            print(f"    -> mu:      {str(tuple(mu.shape)):<20} (Trung bình mỗi chuỗi trong batch)")
            print(f"    -> sigma:   {str(tuple(sigma.shape)):<20} (Độ lệch chuẩn mỗi chuỗi)")
            print(f"    -> y_norm:  {str(tuple(y_norm.shape)):<20} (Chuỗi quá khứ đã khử xu hướng)")
            print("-" * 85)

        # -------------------------------------------------------------------------
        # BƯỚC 2: Chiếu Nén Biến Ngoại Sinh Feature Projection (Section 4.2 Step 1)
        # Công thức: x_tilde_t = FeatureProjection(x_t) in R^{r_tilde}
        # -------------------------------------------------------------------------
        # x_all: [B, L+H, r] -> đưa qua khối ResidualBlock chiếu nén xuống [B, L+H, r_tilde]
        x_proj = self.feature_proj(x_all)                  # [B, L+H, r_tilde]
        
        if verbose:
            print(f"[Bước 2 - Section 4.2 Step 1: Feature Projection]")
            print(f"    Công thức: x_tilde_t = ResidualBlock(x_t) với r -> r_tilde")
            print(f"    -> x_all gốc : {str(tuple(x_all.shape)):<20} (r={self.r} chiều covariates)")
            print(f"    -> x_proj nén: {str(tuple(x_proj.shape)):<20} (r_tilde={self.r_tilde} chiều nén)")
            print("-" * 85)

        # -------------------------------------------------------------------------
        # BƯỚC 3: Ghép Nối Dữ Liệu Đầu Vào Encoder (Section 4.2 Step 2)
        # Công thức: u_enc = [ y_norm ; vec(x_tilde_{1:L+H}) ; a_stat ]
        # -------------------------------------------------------------------------
        flat_x_proj = x_proj.flatten(start_dim=1)          # [B, (L+H) * r_tilde]
        u_enc = torch.cat([y_norm, flat_x_proj, a_stat], dim=-1) # [B, L + (L+H)*r_tilde + r_stat]
        
        if verbose:
            print(f"[Bước 3 - Section 4.2 Step 2: Ghép Nối Đầu Vào Encoder]")
            print(f"    Công thức: u_enc = Concat( y_norm, Flatten(x_proj), a_stat )")
            print(f"    -> flat_x_proj: {str(tuple(flat_x_proj.shape)):<18} ((L+H)*r_tilde = ({self.L}+{self.H})*{self.r_tilde} = {(self.L+self.H)*self.r_tilde})")
            print(f"    -> u_enc hợp nhất: {str(tuple(u_enc.shape)):<16} (L + {(self.L+self.H)*self.r_tilde} + r_stat = {self.L + (self.L+self.H)*self.r_tilde + self.r_stat})")
            print("-" * 85)

        # -------------------------------------------------------------------------
        # BƯỚC 4: Khối Mã Hóa Dày Đặc Dense Encoder (Section 4.2 Step 2)
        # Công thức: e = DenseEncoder(u_enc) in R^{hidden_size}
        # -------------------------------------------------------------------------
        e = self.dense_encoder(u_enc)                      # [B, hidden_size]
        
        if verbose:
            print(f"[Bước 4 - Section 4.2 Step 2: Khối Mã Hóa Dense Encoder]")
            print(f"    Công thức: e = DenseEncoder(u_enc) gồm {len(self.dense_encoder)} khối ResidualBlock")
            print(f"    -> Vector tiềm ẩn e: {str(tuple(e.shape)):<16} (hiddenSize={self.hidden_size})")
            print("-" * 85)

        # -------------------------------------------------------------------------
        # BƯỚC 5: Khối Giải Mã Dày Đặc Dense Decoder (Section 4.3 Step 1)
        # Công thức: g = DenseDecoder(e) in R^{H * p}; D = Reshape(g) in R^{H x p}
        # -------------------------------------------------------------------------
        g = self.dense_decoder(e)                          # [B, H * p]
        D = g.view(-1, self.H, self.p)                     # [B, H, p]
        
        if verbose:
            print(f"[Bước 5 - Section 4.3 Step 1: Khối Giải Mã Dense Decoder]")
            print(f"    Công thức: g = DenseDecoder(e) -> D = Reshape(g, [H, p])")
            print(f"    -> Vector giải mã g:    {str(tuple(g.shape)):<16} (H * p = {self.H} * {self.p} = {self.H * self.p})")
            print(f"    -> Ma trận đặc trưng D: {str(tuple(D.shape)):<16} (Mỗi bước tương lai có 1 vector d_t kích thước p={self.p})")
            print("-" * 85)

        # -------------------------------------------------------------------------
        # BƯỚC 6: Giải Mã Theo Thời Gian Temporal Decoder (Section 4.3 Step 2)
        # Công thức: y_hat_NL(L+t) = TemporalDecoder([ d_t ; x_tilde_{L+t} ])
        # -------------------------------------------------------------------------
        # Lấy nhánh tương lai của covariates đã nén: x_tilde_{L+1 : L+H}
        x_proj_future = x_proj[:, self.L :, :]             # [B, H, r_tilde]
        
        # Ghép nối d_t và x_tilde_{L+t} tại từng bước thời gian
        temp_in = torch.cat([D, x_proj_future], dim=-1)    # [B, H, p + r_tilde]
        
        # Đi qua khối nơ-ron cục bộ sinh ra 1 giá trị vô hướng tại mỗi bước thời gian
        y_hat_nl = self.temporal_decoder(temp_in).squeeze(-1) # [B, H]
        
        if verbose:
            print(f"[Bước 6 - Section 4.3 Step 2: Temporal Decoder & Covariate Highway]")
            print(f"    Công thức: y_hat_NL(L+t) = TemporalDecoder(Concat(d_t, x_tilde_{{L+t}}))")
            print(f"    -> x_proj_future:       {str(tuple(x_proj_future.shape)):<16} (Covariates tương lai L+1 -> L+H)")
            print(f"    -> temp_in ghép nối:    {str(tuple(temp_in.shape)):<16} (p + r_tilde = {self.p} + {self.r_tilde} = {self.p + self.r_tilde})")
            print(f"    -> y_hat_nl phi tuyến:  {str(tuple(y_hat_nl.shape)):<16} (Dự báo thành phần phi tuyến)")
            print("-" * 85)

        # -------------------------------------------------------------------------
        # BƯỚC 7: Nhánh Tắt Tuyến Tính Toàn Cục Global Linear Skip (Section 4.4)
        # Công thức: y_hat_Lin = W_global * y_norm + b_global
        # -------------------------------------------------------------------------
        y_hat_lin = self.global_skip(y_norm)               # [B, H]
        
        if verbose:
            print(f"[Bước 7 - Section 4.4: Nhánh Tắt Tuyến Tính Toàn Cục]")
            print(f"    Công thức: y_hat_lin = W_global * y_norm + b_global (Ánh xạ trực tiếp L -> H)")
            print(f"    -> y_hat_lin tuyến tính: {str(tuple(y_hat_lin.shape)):<16} (Dự báo thành phần xu hướng tuyến tính)")
            print("-" * 85)

        # -------------------------------------------------------------------------
        # BƯỚC 8: Phép Cộng Residual & Khôi Phục Thang Đo RevIN (Section 4.4 & 4.5)
        # Công thức: y_hat_norm = y_hat_nl + y_hat_lin
        #            y_hat_final = y_hat_norm * sigma + mu
        # -------------------------------------------------------------------------
        y_hat_norm = y_hat_nl + y_hat_lin                  # [B, H]
        y_hat_final = y_hat_norm * sigma + mu              # [B, H]
        
        if verbose:
            print(f"[Bước 8 - Section 4.4 & 4.5: Tổng Hợp & Đảo Chuẩn Hóa RevIN]")
            print(f"    Công thức: y_hat_final = (y_hat_nl + y_hat_lin) * sigma + mu")
            print(f"    -> y_hat_norm:          {str(tuple(y_hat_norm.shape)):<16} (Kết quả cộng 2 nhánh trước rescale)")
            print(f"    -> y_hat_final HOÀN TẤT: {str(tuple(y_hat_final.shape)):<16} (Dự báo cuối cùng thang đo gốc!)")
            print("=" * 85)
            print("FORWARD PASS HOÀN TẤT THÀNH CÔNG VỚI TẤT CẢ KÍCH THƯỚC KHỚP HOÀN HẢO!")
            print("=" * 85)

        return y_hat_final


if __name__ == "__main__":
    # Cấu hình siêu tham số đúng theo yêu cầu của người dùng
    B = 32          # Batch size
    L = 720         # Look-back window
    H = 96          # Forecast horizon
    r = 4           # Dynamic covariates gốc
    r_stat = 2      # Static attributes gốc
    r_tilde = 4     # Dynamic covariates nén
    hidden_size = 256
    p = 16
    
    # Khởi tạo mô hình
    torch.manual_seed(42)
    model = TiDEModelSimulation(
        L=L,
        H=H,
        r=r,
        r_stat=r_stat,
        r_tilde=r_tilde,
        hidden_size=hidden_size,
        p=p,
        n_e=2,
        n_d=2,
        temporal_hidden=64,
        dropout=0.1
    )
    model.eval()
    
    # Khởi tạo dữ liệu giả lập
    dummy_y_past = torch.randn(B, L)
    dummy_x_all = torch.randn(B, L + H, r)
    dummy_a_stat = torch.randn(B, r_stat)
    
    # Chạy forward pass
    with torch.no_grad():
        out = model.forward_with_logging(dummy_y_past, dummy_x_all, dummy_a_stat, verbose=True)
    
    # Xác minh shape đầu ra
    assert out.shape == (B, H), f"Shape đầu ra không đúng: {out.shape} != ({B}, {H})"
    print(f"\n[Xác minh Assertion]: Đầu ra đạt chuẩn xác kích thước ({B}, {H})!")
