import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # ต้องเพิ่ม scipy เพื่อคำนวณความน่าจะเป็น

# --- พารามิเตอร์เดิม ---
mean1 = np.array([-3, 5])
mean2 = np.array([3, 5])

cov1 = np.array([[1, 0], [0, 1]]) 
cov2 = np.array([[1, 0], [0, 1]])

pts1 = np.random.multivariate_normal(mean1, cov1, size=100)
pts2 = np.random.multivariate_normal(mean2, cov2, size=100)

# 1. กำหนดพื้นที่กริดสำหรับคำนวณ
x_min, x_max = -10, 10
y_min, y_max = -10, 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))#100คือdata pointเเต่ละเเถว
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 2. คำนวณ Likelihoods ใช้สูตรpdfในสไลหาเส้นเเบ่ง
# L1 = P(x | Class 'a'), L2 = P(x | Class 'b')
L1 = multivariate_normal.pdf(grid_points, mean=mean1, cov=cov1)
L2 = multivariate_normal.pdf(grid_points, mean=mean2, cov=cov2)

# 3. คำนวณ Posterior Probability P(Class 'a' | x)
# P(Class 'a' | x) = L1 / (L1 + L2)
posterior_a = L1 / (L1 + L2)
posterior_a = posterior_a.reshape(xx.shape)

# --- โค้ด Plot กราฟ ---
plt.figure(figsize=(8, 8))
plt.scatter(pts1[:, 0], pts1[:, 1], marker='.', s=50, alpha=0.5, color='red', label = 'a')
plt.scatter(pts2[:, 0], pts2[:, 1], marker='.', s=50, alpha=0.5, color='blue', label = 'b')

# 4. พล็อตเส้นแบ่ง (Decision Boundary)
# เส้นแบ่งคือเส้นที่ Posterior Probability = 0.5 คือครั่งนึงมากกว่า0.5ไปa
plt.contour(xx, yy, posterior_a, levels=[0.5], colors='Brown', linestyles='-', linewidths=2) 
# 

plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.legend()
plt.grid()
plt.title('Data with Linear Decision Boundary (LDA)')
plt.show()