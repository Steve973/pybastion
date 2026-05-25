class SyntheticDependency:
    prop: str

    def transform(self, value: str) -> str:
        return self.prop + ":" + value.upper()
