import matplotlib.pyplot as plt
import seaborn as sns

def show_pairwise_relations(pairs):
    plt.figure(figsize=(6,3))
    sns.pairplot(pairs, vars=['coint_pvalue','norm_price_dist','spread_vol','half_life'])
    plt.suptitle("Pairwise Relationships Between Metrics", y=1.02)
    plt.show()
