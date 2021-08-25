            # do binarize
            if self.config["Global"].get("feature_binarize") == "round":
                batch_feas = paddle.round(batch_feas).astype("float32") * 2.0 - 1.0

            if self.config["Global"].get("feature_binarize") == "sign":
                batch_feas = paddle.sign(batch_feas).astype("float32")
