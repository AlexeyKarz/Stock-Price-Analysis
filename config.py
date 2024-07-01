class Config:
    DEBUG = False
    TESTING = False
    API_KEY = 'NSQ25HG8ERO35TPU'


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    pass
