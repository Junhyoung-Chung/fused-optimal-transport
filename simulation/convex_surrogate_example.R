library(ggplot2)
library(showtext)    # LaTeX-style font
library(dplyr)
library(tidyr)

# ---- 1) 좌표 정의 ----
X <- tibble::tibble(
  id = paste0("x", 1:3),
  i  = 1:3,
  x  = c(4, 2, 5),
  y  = c(8, 4, 6)
)

# Y_1 = X_1 + (4,0),  Y_2 = X_3 + (4,0),  Y_3 = X_2 + (4,0)
Y <- tibble::tibble(
  id = paste0("y", c(1,2,3)),
  i  = c(1,2,3),
  x  = X$x[c(1,3,2)] + 4,
  y  = X$y[c(1,3,2)] + c(0, -0.7, 0)
)

# ---- 2) 쌍별 거리 ----
dist_pair <- function(df) {
  d12 <- sqrt((df$x[1]-df$x[2])^2 + (df$y[1]-df$y[2])^2)
  d13 <- sqrt((df$x[1]-df$x[3])^2 + (df$y[1]-df$y[3])^2)
  d23 <- sqrt((df$x[2]-df$x[3])^2 + (df$y[2]-df$y[3])^2)
  c(d12=d12, d13=d13, d23=d23)
}
dx <- dist_pair(X)
dy <- dist_pair(Y)

x12 <- dx["d12"]; x13 <- dx["d13"]; x23 <- dx["d23"]
y12 <- dy["d12"]; y13 <- dy["d13"]; y23 <- dy["d23"]

# ---- 3) 계수 계산 (일반식) ----
A <- 2 * ((x12 - y12)^2 + (x13 - y13)^2 + (x23 - y23)^2)
B <- 2 * ((x13 - y12)^2 + (x12 - y13)^2 + (x23 - y23)^2)
C <- (x12 + x13 - y12 - y13)^2

a <- A + B - 2*C
b <- -2*B + 2*C
c <- B

# GW trace 항의 계수 (상수항은 필요 없음)
D <- 2 * (x12*y12 + x13*y13 + x23*y23)
E <- 2 * (x13*y12 + x12*y13 + x23*y23)
G <- 2 * (x12 + x13) * (y12 + y13)

# ---- 4) t-그리드에서 곡선 값 ----
tgrid <- tibble::tibble(t = seq(0, 1, length.out = 400))

convex_df <- tgrid %>%
  mutate(cs = (a*t^2 + b*t + c)/9)

# GW penalty는 const - (2/3)*(D t^2 + E (1-t)^2 + F t(1-t))
# 모양만 비교하려고 const는 생략
gw_df <- tgrid %>%
  mutate(gw = -(2/3) * (D*t^2 + E*(1-t)^2 + G*t*(1-t)))


# ---- 6) (b) Convex surrogate plot ----

# ---- 2) minimizer 계산 ----
# ===== 폰트 기본값(안전) =====

# minimizer
t_cs_star <- (B - C) / ((A - C) + (B - C))

library(ggplot2)
library(dplyr)

# ---- 두 데이터프레임 결합 ----
plot_df <- bind_rows(
  convex_df %>% mutate(type = "Proposed convex surrogate", value = cs),
  gw_df %>% mutate(type = "GW penalty (up to const.)", value = gw + 51)
)

# ---- 한 그래프에 합치기 ----
p_all <-
  ggplot(plot_df, aes(x = t, y = value, color = type, linetype = type)) +
  geom_line(linewidth = 1.5) +
  geom_vline(xintercept = t_cs_star, linetype = "dashed", color = "red", linewidth = 0.7) +
  annotate("text",
           x = min(1, t_cs_star + 0.04),
           y = max(convex_df$cs) * 7.5,
           label = "t[CS]^'*'", parse = TRUE,
           size = 7) +
  labs(x = "t", y = NULL, color = NULL, linetype = NULL) +
  theme_classic(base_size = 20) +
  scale_x_continuous(breaks = c(0, 0.5, 1)) +
  theme(
    legend.position = c(0.75, 0.95),   # 우측 상단 (좌표 0~1)
    legend.background = element_blank(),
    legend.box.background = element_rect(color = NA),
    legend.key.width = unit(1.8, "cm"),
    axis.title.x = element_text(size = 18),
    axis.text  = element_text(size = 14),
    plot.margin = margin(8, 8, 8, 8)
  ) +
  scale_color_manual(values = c("Proposed convex surrogate" = "black",
                                "GW penalty (up to const.)" = "gray40")) +
  scale_linetype_manual(values = c("Proposed convex surrogate" = "solid",
                                   "GW penalty (up to const.)" = "dotted"))

# 결과 출력
p_all

ggsave(
  filename = "convex_vs_gw.png",
  plot = p_all,
  width = 8,        # 가로 (inch) — 논문용은 6~8이 이상적
  height = 5,       # 세로 (inch)
  units = "in",
  dpi = 600,        # 해상도: 300(기본)~600(출판용)
  bg = "white"      # 배경 흰색 (투명하면 LaTeX에 따라 깨질 수 있음)
)

