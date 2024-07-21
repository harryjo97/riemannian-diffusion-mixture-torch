from geomstats.geometry.product_manifold import ProductSameManifold
from geomstats.geometry.hypersphere import Hypersphere


class Torus(ProductSameManifold):
    def __init__(self, dim, **kwargs):
        super(Torus, self).__init__(Hypersphere(1), dim, **kwargs)
