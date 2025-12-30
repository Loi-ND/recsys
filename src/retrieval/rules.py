import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict
from tqdm import tqdm
import math
import implicit
from scipy.sparse import coo_matrix

# * scores of rules are the bigger the better


class PersonalRetrieveRule(ABC):
    """Use certain rules to respectively retrieve items for each customer."""

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items

        Returns:
            pd.DataFrame: (customer_id, article_id, method, score)
        """


class UserGroupRetrieveRule(ABC):
    """Use certain rules to respectively retrieve items for each group of customers."""

    def merge(self, result: pd.DataFrame):
        result = result[[*self.cat_cols, self.iid, "method", "score"]]

        user = self.data["user"][[*self.cat_cols, "customer_id"]]
        tmp_df = pd.DataFrame({"customer_id": self.customer_list})
        tmp_df = tmp_df.merge(user, on="customer_id", how="left")
        tmp_df = tmp_df.merge(result, on=[*self.cat_cols], how="left")

        tmp_df = tmp_df[["customer_id", *self.cat_cols, self.iid, "score", "method"]]

        return tmp_df

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items

        Returns:
            pd.DataFrame: (*group_cols, article_id, method, score)
        """


class ItemGroupRetrieveRule(ABC):
    """Use certain rules to respectively retrieve items in each group of items."""

    def merge(self, top_items: pd.DataFrame):
        """
        top_items gồm:
            cat_cols + [item_id, score, method]
        """

        # 1. Lấy transaction để map user -> item -> cat_cols
        trans = self.trans_df[["customer_id", self.iid] + list(self.cat_cols)].drop_duplicates()

        # 2. Map user -> tất cả các group cat_cols mà họ đã mua
        user_group = trans[["customer_id"] + list(self.cat_cols)].drop_duplicates()

        # 3. Lọc theo danh sách khách hàng cần recommend
        user_group = user_group[user_group["customer_id"].isin(self.customer_list)]

        # 4. Join user_group với top_items theo cat_cols => user chỉ nhận item nhóm mình mua
        result = user_group.merge(top_items, on=list(self.cat_cols), how="left")

        # 5. Chỉ giữ cột cần thiết
        result = result[["customer_id"] + [self.iid, "score", "method"]]

        return result
    
    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items

        Returns:
            pd.DataFrame: (*group_cols, article_id, method, score)
        """

class GlobalRetrieveRule(ABC):
    """Use certain rules to retrieve items for all customers."""

    def merge(self, result: pd.DataFrame):
        result = result[[self.iid, "method", "score"]]

        num_item = result.shape[0]
        num_user = self.customer_list.shape[0]

        tmp_user = np.repeat(self.customer_list, num_item)
        tmp_df = result.iloc[np.tile(np.arange(num_item), num_user)]
        tmp_df = tmp_df.reset_index(drop=True)
        tmp_df["customer_id"] = tmp_user

        return tmp_df

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items

        Returns:
            pd.DataFrame: (article_id, method, score)
        """


class FilterRule(ABC):
    """Use certain rules to remove some retrieved items."""

    @abstractmethod
    def retrieve(self) -> List:
        """Retrieve items

        Returns:
            List: items to be removed
        """


# * ======================= Personal Retrieve Rules ======================= *


class OrderHistory(PersonalRetrieveRule):
    """Retrieve recently bought items by the customer."""

    def __init__(
        self,
        trans_df: pd.DataFrame,
        days: int = 7,
        n: int = None,
        name: str = "1",
        item_id: str = "article_id",
    ):
        """Initialize OrderHistory.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        days : int, optional
            Length of time window when getting user buying history, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``None``.
        name : str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.trans_df = trans_df[["t_dat", "customer_id", item_id]]
        self.days = days
        self.top_k = n
        self.name = name

    def retrieve(self) -> pd.DataFrame:
        df = self.trans_df.reset_index()
        df["t_dat"] = pd.to_datetime(df["t_dat"])

        tmp = df.groupby("customer_id").t_dat.max().reset_index()
        tmp.columns = ["customer_id", "max_dat"]
        res = df.merge(tmp, on=["customer_id"], how="left")

        res["diff_dat"] = (res.max_dat - res.t_dat).dt.days
        res = res.loc[res["diff_dat"] < self.days].reset_index(drop=True)

        res = res.sort_values(by=["diff_dat"], ascending=True).reset_index(drop=True)
        res = res.groupby(["customer_id", self.iid], as_index=False).first()

        res = res.reset_index()
        res = res.sort_values(by="index", ascending=False).reset_index(drop=True)
        res["rank"] = res.groupby(["customer_id"])["index"].rank(
            ascending=True, method="first"
        )
        res["score"] = -res["diff_dat"]

        if self.top_k is not None:
            res = res.loc[res["rank"] <= self.top_k]

        res["method"] = f"OrderHistory_{self.name}"
        res = res[["customer_id", self.iid, "score", "method"]]

        return res


class OrderHistoryDecay(PersonalRetrieveRule):
    """Retrieve recently bought items by the customer with decay."""

    def __init__(
        self,
        trans_df: pd.DataFrame,
        days: int = 7,
        n: int = None,
        name: str = "1",
        item_id: str = "article_id",
    ):
        """Initialize OrderHistoryDecay.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        days : int, optional
            Length of time window when getting user buying history, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``None``.
        name : str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat"]]
        self.name = name
        self.days = days
        self.n = n

    def retrieve(self):
        
        df = self.trans_df.copy()
        df["t_dat"] = pd.to_datetime(df["t_dat"])

        # ================
        # 1. Basic fields
        # ================
        last_ts = df["t_dat"].max()

        df["dat_gap"] = (last_ts - df["t_dat"]).dt.days
        df["dat_gap"] = df["dat_gap"].clip(lower=1)

        # ==============================
        # 2. Tính last_day theo window D
        # ==============================
        df["last_day"] = df["t_dat"].dt.floor(f"{self.days}D")

        # last day của toàn dataset
        last_day = df["last_day"].max()

        # ===============================
        # 3. Tính period sale cho mỗi item
        # ===============================
        period_sales = (
            df.groupby(["last_day", self.iid])["t_dat"]
            .count()
            .rename("period_sale")
            .reset_index()
        )

        # ===============================
        # 4. Merge period sale theo từng bản ghi
        # ===============================
        df = df.merge(
            period_sales,
            on=["last_day", self.iid],
            how="left"
        )

        # ===========================================
        # 5. Lấy period_sale của ngày cuối để so sánh
        # ===========================================
        last_ps = period_sales[period_sales["last_day"] == last_day][
            [self.iid, "period_sale"]
        ].rename(columns={"period_sale": "period_sale_targ"})

        df = df.merge(last_ps, on=self.iid, how="left")
        df["period_sale_targ"] = df["period_sale_targ"].fillna(0)

        # ========================
        # 6. Compute sale quotient
        # ========================
        df["quotient"] = df["period_sale_targ"] / df["period_sale"].replace(0, np.nan)
        df["quotient"] = df["quotient"].fillna(0)

        # ==============================
        # 7. Decay value theo dat_gap
        # ==============================
        x = df["dat_gap"]
        a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3

        df["value"] = a / np.sqrt(x) + b * np.exp(-c * x) - d
        df["value"] = df["value"].clip(lower=0)
        df["value"] *= df["quotient"]

        # ==============================
        # 8. Aggregate theo customer - item
        # ==============================
        df = (
            df.groupby(["customer_id", self.iid], as_index=False)["value"]
            .sum()
        )

        # ==============================
        # 9. Ranking & filtering
        # ==============================
        df = df.sort_values(by="value", ascending=False)
        df["rank"] = df.groupby("customer_id")["value"].rank(
            ascending=False, method="first"
        )

        df["score"] = df["value"]

        df = df[df["value"] > 150]

        if self.n is not None:
            df = df[df["rank"] <= self.n]

        df["method"] = f"OrderHistoryDecay_{self.name}"

        return df[["customer_id", self.iid, "score", "method"]]


class ItemPair(PersonalRetrieveRule):
    """Customers who bought this often bought this."""

    def __init__(
        self, trans_df: pd.DataFrame, name: str = "1", item_id: str = "article_id"
    ):
        """Initialize ItemPair.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        name: str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid]].drop_duplicates()
        self.name = name

    def _get_freq_pair(self) -> pd.DataFrame:
        """Generate dict of frequent item pairs in target transaction dataframe.

        Returns
        -------
        pd.DataFrame
            Frequent item pairs.
        """
        df = self.trans_df
        df2 = df.rename(columns={self.iid: "pair"})

        pair = df.merge(df2, on="customer_id")
        pair = pair[pair[self.iid] != pair["pair"]]
        pair["count"] = 1
        pair = pair.groupby([self.iid, "pair"])["count"].sum().reset_index()
        pair = pair.sort_values("count", ascending=False).reset_index(drop=True)
        pair = pair.groupby(self.iid).first().reset_index()
        pair = pair.sort_values(by="count", ascending=False).reset_index(drop=True)
        pair["score"] = pair["count"]

        return pair

    def retrieve(self) -> pd.DataFrame:
        pair = self._get_freq_pair()

        df = self.trans_df
        df = df.merge(pair, on=self.iid, how="left")

        df = df.loc[df["pair"].notnull()]
        df = df.drop_duplicates(["customer_id", "pair"])

        df[self.iid] = df["pair"].astype("int32")
        df["method"] = "ItemPairRetrieve_" + self.name

        df = df[["customer_id", self.iid, "method", "score"]]
        return df


class ALS(PersonalRetrieveRule):
    """ALS Collaborative Filtering."""

    def __init__(
        self,
        customer_list: List,
        trans_df: pd.DataFrame,
        days: int = 5,
        n: int = 12,
        name: str = "1",
        item_id: str = "article_id",
        num_items: int = 105542,
        factors: int = 200,
        iter_num: int = 15,
        item_pool: List = None,
    ):
        """Initialize ALS.

        Parameters
        ----------
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        n : int, optional
            Get top `n` recently bought items, by default ``12``.
        name: str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        num_items: int, optional
            Number of unique items in the whole dataset, by default ``105542``.
        """
        self.customer_list = customer_list
        self.iid = item_id
        self.n = n
        self.trans_df = trans_df[["customer_id", self.iid]].drop_duplicates()
        self.name = name
        self.num_items = num_items
        self.factors = factors
        self.iter_num = iter_num

        if item_pool is not None:
            self.target_items = item_pool
        else:
            df = trans_df[["article_id", "t_dat"]]
            df["t_dat"] = pd.to_datetime(df["t_dat"])
            df = df[df["t_dat"] > df["t_dat"].max() - pd.Timedelta(days=days)]
            self.target_items = df["article_id"].value_counts().index[:1000]

    def _to_user_item_coo(self, df: pd.DataFrame):
        """Turn a dataframe with transactions into a COO sparse items x users matrix"""
        row = df["customer_id"].values - 1  # * input id starts from 1
        col = df[self.iid].values - 1  # * input id starts from 1
        data = np.ones(df.shape[0])
        coo = coo_matrix((data, (row, col)), shape=(1371980, self.num_items))
        return coo

    def _train(self, coo_train, reg=0.01, verbose=False):
        model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iter_num,
            # regularization=reg,
            random_state=42,
        )
        model.fit(coo_train, show_progress=verbose)
        return model

    def _predict(self, model, uids, coo_train):
        preds = np.zeros((uids.shape[0], self.n))
        scores = np.zeros((uids.shape[0], self.n))
        batch_size = 10000
        for startidx in range(0, len(uids), batch_size):
            batch = uids[startidx : startidx + batch_size] - 1
            ids, batch_scores = model.recommend(
                batch,
                coo_train[batch],
                N=self.n,
                items=self.target_items - 1,
                filter_already_liked_items=False,
            )
            preds[startidx : startidx + batch_size] = ids + 1
            scores[startidx : startidx + batch_size] = batch_scores

        candidates = pd.DataFrame(
            {
                "customer_id": np.repeat(uids, self.n),
                "article_id": preds.reshape(-1),
                "method": "ALS_" + self.name,
                "score": scores.reshape(-1),
            }
        )

        return candidates

    def retrieve(self) -> pd.DataFrame:
        coo = self._to_user_item_coo(self.trans_df)
        coo = coo.tocsr()

        model = self._train(coo)
        candidates = self._predict(model, self.customer_list, coo)
        return candidates


class UserGroupALS(PersonalRetrieveRule):
    """ALS Collaborative Filtering for each group of users."""

    def __init__(
        self,
        data: Dict,
        customer_list: List,
        trans_df: pd.DataFrame,
        cat_col: str,
        days: int = 5,
        n: int = 12,
        name: str = "1",
        item_id: str = "article_id",
        num_items: int = 105542,
        factors: int = 200,
        iter_num: int = 15,
    ):
        """Initialize UserGroupALS.

        Parameters
        ----------
        data : Dict 
            Dict of dataframes.
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        cat_col: str
            Name of user group column.
        n : int, optional
            Get top `n` recently bought items, by default ``12``.
        name: str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        num_items: int, optional
            Number of unique items in the whole dataset, by default ``105542``.
        """
        self.data = data
        self.customer_list = customer_list
        self.iid = item_id
        self.n = n
        if cat_col not in trans_df.columns:
            self.trans_df = trans_df[["customer_id", self.iid, "t_dat"]]
            user_info = data["user"][["customer_id", cat_col]]
            self.trans_df = self.trans_df.merge(user_info, on="customer_id", how="left")
        else:
            self.trans_df = trans_df[["customer_id", self.iid, cat_col, "t_dat"]]

        self.trans_df = self.trans_df.drop_duplicates(
            ["customer_id", self.iid, cat_col]
        )
        self.cat_col = cat_col
        self.name = name
        self.num_items = num_items
        self.factors = factors
        self.iter_num = iter_num

        self.target_items = {}
        for value in self.data["user"][cat_col].unique():
            df = self.trans_df[["article_id", "t_dat", cat_col]]
            df = df[df[cat_col] == value]
            df["t_dat"] = pd.to_datetime(df["t_dat"])
            df = df[df["t_dat"] > df["t_dat"].max() - pd.Timedelta(days=days)]
            self.target_items[value] = df["article_id"].value_counts().index[:1000]

    def _to_user_item_coo(self, df: pd.DataFrame):
        """Turn a dataframe with transactions into a COO sparse items x users matrix"""
        row = df["customer_id"].values - 1  # * input id starts from 1
        col = df[self.iid].values - 1  # * input id starts from 1
        data = np.ones(df.shape[0])
        coo = coo_matrix((data, (row, col)), shape=(1371980, self.num_items))
        return coo

    def _train(self, coo_train, factors=200, iter_num=3, reg=0.01, verbose=False):
        model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iter_num,
            # regularization=reg,
            random_state=42,
        )
        model.fit(coo_train, show_progress=verbose)
        return model

    def _predict(self, model, uids, coo_train, value):
        preds = np.zeros((uids.shape[0], self.n))
        scores = np.zeros((uids.shape[0], self.n))
        batch_size = 10000
        for startidx in range(0, len(uids), batch_size):
            batch = uids[startidx : startidx + batch_size] - 1
            ids, batch_scores = model.recommend(
                batch,
                coo_train[batch],
                N=self.n,
                items=self.target_items[value] - 1,
                filter_already_liked_items=False,
            )
            preds[startidx : startidx + batch_size] = ids + 1
            scores[startidx : startidx + batch_size] = batch_scores

        candidates = pd.DataFrame(
            {
                "customer_id": np.repeat(uids, self.n),
                "article_id": preds.reshape(-1),
                "method": "UGALS_" + self.name,
                "score": scores.reshape(-1),
            }
        )

        return candidates

    def retrieve(self) -> pd.DataFrame:
        res = None
        user_info = pd.DataFrame({"customer_id": self.customer_list})
        user_info = user_info.merge(
            self.data["user"][["customer_id", self.cat_col]],
            on="customer_id",
            how="left",
        )
        for value in self.trans_df[self.cat_col].unique():
            tmp_df = self.trans_df[self.trans_df[self.cat_col] == value]
            coo = self._to_user_item_coo(tmp_df)
            coo = coo.tocsr()
            model = self._train(coo)
            uids = user_info.loc[user_info[self.cat_col] == value, "customer_id"].values
            candidates = self._predict(model, uids, coo, value)
            res = pd.concat([res, candidates], ignore_index=True)

        return res

  # nếu bạn dùng base class này


class UserGroupLightFM(PersonalRetrieveRule):
    """
    LightFM (WARP/BPR) collaborative filtering cho từng group người dùng.
    """

    def __init__(
        self,
        data: Dict,
        customer_list: List,
        trans_df: pd.DataFrame,
        cat_col: str,
        days: int = 5,
        n: int = 12,
        name: str = "1",
        item_id: str = "article_id",
        num_items: int = 105542,
        loss: str = "warp",          # warp, bpr, logistic, hinge
        no_components: int = 64,
        epochs: int = 15,
    ):
        self.data = data
        self.customer_list = np.array(customer_list)
        self.iid = item_id
        self.n = n
        self.name = name
        self.num_items = num_items
        self.loss = loss
        self.no_components = no_components
        self.epochs = epochs
        self.cat_col = cat_col

        # Chuẩn hoá trans_df
        if cat_col not in trans_df.columns:
            df = trans_df[["customer_id", item_id, "t_dat"]]
            user_info = data["user"][["customer_id", cat_col]]
            self.trans_df = df.merge(user_info, on="customer_id", how="left")
        else:
            self.trans_df = trans_df[["customer_id", item_id, cat_col, "t_dat"]]

        self.trans_df = self.trans_df.drop_duplicates(["customer_id", item_id, cat_col])

        # Lấy các target items gần đây mỗi group
        self.target_items = {}
        for value in self.data["user"][cat_col].unique():
            df = self.trans_df[self.trans_df[cat_col] == value].copy()
            df["t_dat"] = pd.to_datetime(df["t_dat"])
            df = df[df["t_dat"] > df["t_dat"].max() - pd.Timedelta(days=days)]
            self.target_items[value] = df[item_id].value_counts().index[:1000].values


    # ====== Build sparse matrix ======
    def _to_coo(self, df: pd.DataFrame):
        """
        Build user-item interaction matrix for LightFM.
        user_id và item_id phải >= 0
        """
        # Convert id → 0-based
        df_user = df["customer_id"].values - 1
        df_item = df[self.iid].values - 1
        data = np.ones(df.shape[0])

        num_users = self.data["user"]["customer_id"].max()
        mat = coo_matrix(
            (data, (df_user, df_item)),
            shape=(num_users, self.num_items),
        )
        return mat


    # ===== Training LightFM =====
    def _train(self, coo_train):
        model = LightFM(
            no_components=self.no_components,
            loss=self.loss,
            random_state=42,
        )
        model.fit(coo_train, epochs=self.epochs, num_threads=4)
        return model


    # ===== Run prediction for group =====
    def _predict(self, model, uids, coo_train, value):
        """
        Predict top-n for given user ids in this group.
        """
        preds = []
        scores = []

        items = (self.target_items[value] - 1).astype(np.int32)  # convert to 0-based

        for uid in uids:
            uid0 = uid - 1

            score_vec = model.predict(uid0, items)
            rank_idx = np.argsort(-score_vec)[: self.n]

            preds.append(items[rank_idx] + 1)
            scores.append(score_vec[rank_idx])

        preds = np.array(preds)
        scores = np.array(scores)

        return pd.DataFrame({
            "customer_id": np.repeat(uids, self.n),
            "article_id": preds.reshape(-1),
            "method": "UGLFM_" + self.name,
            "score": scores.reshape(-1),
        })


    # ===== Main retrieval =====
    def retrieve(self) -> pd.DataFrame:
        res = None

        user_info = pd.DataFrame({"customer_id": self.customer_list})
        user_info = user_info.merge(
            self.data["user"][["customer_id", self.cat_col]],
            on="customer_id",
            how="left",
        )

        for value in self.trans_df[self.cat_col].unique():
            print(f"Training LightFM group = {value}")

            df = self.trans_df[self.trans_df[self.cat_col] == value]

            # Sparse matrix
            coo = self._to_coo(df)

            # Train model
            model = self._train(coo)

            # Predict for users in group
            uids = user_info[user_info[self.cat_col] == value]["customer_id"].values
            if len(uids) == 0:
                continue

            candidates = self._predict(model, uids, coo, value)
            res = pd.concat([res, candidates], ignore_index=True)

        return res


class BPR(PersonalRetrieveRule):
    """BPR Collaborative Filtering."""

    def __init__(
        self,
        customer_list: List,
        trans_df: pd.DataFrame,
        days: int = 5,
        n: int = 12,
        name: str = "1",
        item_id: str = "article_id",
        num_items: int = 105542,
        factors: int = 300,
        iter_num: int = 15,
        item_pool: List = None,
    ):
        """Initialize BPR.

        Parameters
        ----------
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        n : int, optional
            Get top `n` recently bought items, by default ``12``.
        name: str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        num_items: int, optional
            Number of unique items in the whole dataset, by default ``105542``.
        """
        self.customer_list = customer_list
        self.iid = item_id
        self.n = n
        self.trans_df = trans_df[["customer_id", self.iid]].drop_duplicates()
        self.name = name
        self.num_items = num_items
        self.factors = factors
        self.iter_num = iter_num

        if item_pool is not None:
            self.target_items = item_pool
        else:
            df = trans_df[["article_id", "t_dat"]]
            df["t_dat"] = pd.to_datetime(df["t_dat"])
            df = df[df["t_dat"] > df["t_dat"].max() - pd.Timedelta(days=days)]
            self.target_items = df["article_id"].value_counts().index[:1000]

    def _to_user_item_coo(self, df: pd.DataFrame):
        """Turn a dataframe with transactions into a COO sparse items x users matrix"""
        row = df["customer_id"].values - 1  # * input id starts from 1
        col = df[self.iid].values - 1  # * input id starts from 1
        data = np.ones(df.shape[0])
        coo = coo_matrix((data, (row, col)), shape=(1371980, self.num_items))
        return coo

    def _train(self, coo_train, reg=0.01, verbose=False):
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            iterations=self.iter_num,
            learning_rate=0.05,
            # regularization=reg,
            random_state=42,
        )
        model.fit(coo_train, show_progress=verbose)
        return model

    def _predict(self, model, uids, coo_train):
        preds = np.zeros((uids.shape[0], self.n))
        scores = np.zeros((uids.shape[0], self.n))
        batch_size = 10000
        for startidx in range(0, len(uids), batch_size):
            batch = uids[startidx : startidx + batch_size] - 1
            ids, batch_scores = model.recommend(
                batch,
                coo_train[batch],
                N=self.n,
                items=self.target_items - 1,
                filter_already_liked_items=False,
            )
            preds[startidx : startidx + batch_size] = ids + 1
            scores[startidx : startidx + batch_size] = batch_scores

        candidates = pd.DataFrame(
            {
                "customer_id": np.repeat(uids, self.n),
                "article_id": preds.reshape(-1),
                "method": "BPR_" + self.name,
                "score": scores.reshape(-1),
            }
        )

        return candidates

    def retrieve(self) -> pd.DataFrame:
        coo = self._to_user_item_coo(self.trans_df)
        coo = coo.tocsr()

        model = self._train(coo)
        candidates = self._predict(model, self.customer_list, coo)

        return candidates


class UserGroupBPR(PersonalRetrieveRule):
    """ALS Collaborative Filtering for each group of users."""

    def __init__(
        self,
        data: Dict,
        customer_list: List,
        trans_df: pd.DataFrame,
        cat_col: str,
        days: int = 5,
        n: int = 12,
        name: str = "1",
        item_id: str = "article_id",
        num_items: int = 105542,
        factors: int = 300,
        iter_num: int = 15,
    ):
        """Initialize UserGroupBPR.

        Parameters
        ----------
        data : Dict
            Dict of dataframes.
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        cat_col: str
            Name of user group column.
        n : int, optional
            Get top `n` recently bought items, by default ``12``.
        name: str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        num_items: int, optional
            Number of unique items in the whole dataset, by default ``105542``.
        """
        self.data = data
        self.customer_list = customer_list
        self.iid = item_id
        self.n = n
        self.trans_df = trans_df[["customer_id", self.iid, cat_col]].drop_duplicates()
        self.cat_col = cat_col
        self.name = name
        self.num_items = num_items
        self.factors = factors
        self.iter_num = iter_num

        df = trans_df[["article_id", "t_dat"]]
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        df = df[df["t_dat"] > df["t_dat"].max() - pd.Timedelta(days=days)]
        self.target_items = df["article_id"].value_counts().index[:1000]

    def _to_user_item_coo(self, df: pd.DataFrame):
        """Turn a dataframe with transactions into a COO sparse items x users matrix"""
        row = df["customer_id"].values - 1  # * input id starts from 1
        col = df[self.iid].values - 1  # * input id starts from 1
        data = np.ones(df.shape[0])
        coo = coo_matrix((data, (row, col)), shape=(1371980, self.num_items))
        return coo

    def _train(self, coo_train, factors=300, iter_num=300, reg=0.01, verbose=False):
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            iterations=self.iter_num,
            learning_rate=0.05,
            # regularization=reg,
            random_state=42,
        )
        model.fit(coo_train, show_progress=verbose)
        return model

    def _predict(self, model, uids, coo_train):
        preds = np.zeros((uids.shape[0], self.n))
        scores = np.zeros((uids.shape[0], self.n))
        batch_size = 10000
        for startidx in range(0, len(uids), batch_size):
            batch = uids[startidx : startidx + batch_size] - 1
            ids, batch_scores = model.recommend(
                batch,
                coo_train[batch],
                N=self.n,
                items=self.target_items - 1,
                filter_already_liked_items=False,
            )
            preds[startidx : startidx + batch_size] = ids + 1
            scores[startidx : startidx + batch_size] = batch_scores

        candidates = pd.DataFrame(
            {
                "customer_id": np.repeat(uids, self.n),
                "article_id": preds.reshape(-1),
                "method": "UGBPR_" + self.name,
                "score": scores.reshape(-1),
            }
        )

        return candidates

    def retrieve(self) -> pd.DataFrame:
        res = None
        user_info = pd.DataFrame({"customer_id": self.customer_list})
        user_info = user_info.merge(
            self.data["user"][["customer_id", self.cat_col]],
            on="customer_id",
            how="left",
        )
        for value in self.trans_df[self.cat_col].unique():
            tmp_df = self.trans_df[self.trans_df[self.cat_col] == value]
            coo = self._to_user_item_coo(tmp_df)
            coo = coo.tocsr()
            model = self._train(coo)
            uids = user_info.loc[user_info[self.cat_col] == value, "customer_id"].values
            candidates = self._predict(model, uids, coo)
            res = pd.concat([res, candidates], ignore_index=True)

        return res


class ItemCF(PersonalRetrieveRule):
    """Item-Item Collaborative Filtering."""

    def __init__(
        self, history_df: pd.DataFrame, target_df: pd.DataFrame, top_k=20, name="1"
    ):
        """Initialize ItemCF.

        Parameters
        ----------
        history_df : pd.DataFrame
            Dataframe of transaction records.
        target_df : pd.DataFrame
            Dataframe of target customer transaction records.
        top_k : int, optional
            Get top `k` similar items, by default ``20``.
        name: str, optional
            Name of the rule, by default ``1``.
        """

        self.history_df = history_df.groupby("customer_id")["article_id"].apply(list)
        self.target_df = target_df.groupby("customer_id")["article_id"].apply(list)
        self.top_k = top_k
        self.name = name

    def _item_cf(self, top_k=20):
        sim_item = {}
        item_cnt = {}

        for user in self.history_df.keys():
            items = self.history_df[user]
            for i, item in enumerate(items[:-1]):
                sim_item.setdefault(item, {})
                item_cnt[item] = item_cnt.get(item, 0) + 1
                for j, relate_item in enumerate(items):
                    if i == j:
                        continue
                    loc_alpha = 1.0 if j > i else 0.9  # * direction factor
                    loc_weight = loc_alpha * 0.7 ** (
                        abs(j - i) - 1
                    )  # * distance factor
                    sim_item[item].setdefault(relate_item, 0)
                    sim_item[item][relate_item] += loc_weight / math.log(
                        1 + len(items)
                    )  # * popularity factor

        sim_item_corr = sim_item.copy()
        for i, related_items in sim_item.items():
            sim_item_corr[i] = sorted(
                related_items.items(), key=lambda x: x[1], reverse=True
            )

        pred_next = {}
        pred_score = {}
        for item in sim_item_corr:
            if len(sim_item_corr[item]) >= 5:
                l = pred_next.get(item, [])
                l2 = pred_score.get(item, [])
                relate_items = sim_item_corr[item][:top_k]
                pred_next[item] = l + [x[0] for x in relate_items]
                pred_score[item] = l2 + [x[1] for x in relate_items]

        return pred_next, pred_score

    def _predict(self, item_dict, score_dict):
        item_cf_res = {}
        item_cf_score = {}
        for user in self.target_df.keys():
            l = item_cf_res.get(user, [])
            l2 = item_cf_score.get(user, [])
            for item in self.target_df[user]:
                if item in item_dict:
                    l += item_dict[item]
                    l2 += score_dict[item]
            item_cf_res[user] = l
            item_cf_score[user] = l2

        length = sum([len(item_cf_res[x]) for x in item_cf_res])
        candidates = np.zeros((length, 3))
        p = 0
        for uid in item_cf_res:
            tmp_items = item_cf_res[uid]
            tmp_scores = item_cf_score[uid]
            if len(tmp_items) == 0:
                continue
            candidates[p : p + len(tmp_items), 0] = uid
            candidates[p : p + len(tmp_items), 1] = tmp_items
            candidates[p : p + len(tmp_items), 2] = tmp_scores
            p += len(tmp_items)
        candidates = pd.DataFrame(
            candidates[:p], columns=["customer_id", "article_id", "score"]
        )
        candidates["method"] = "ItemCF_" + self.name
        return candidates

    def retrieve(self, top_k=20):
        item_dict, score_dict = self._item_cf(top_k)
        candidates = self._predict(item_dict, score_dict)
        return candidates


class UserGroupItemCF(PersonalRetrieveRule):
    """Item-Item Collaborative Filtering for a user group."""

    def __init__(
        self,
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
        cat_col: str,
        top_k=20,
        name="1",
    ):
        """Initialize UserGroupItemCF.

        Parameters
        ----------
        history_df : pd.DataFrame
            Dataframe of transaction records.
        target_df : pd.DataFrame
            Dataframe of target customer transaction records.
        cat_col : str
            Column name of user group.
        top_k : int, optional
            Get top `k` similar items, by default ``20``.
        name : str, optional
            Name of the rule, by default ``1``.
        """
        self.history_dict = {}
        self.target_dict = {}
        for value in history_df[cat_col].unique():
            self.history_dict[value] = history_df[history_df[cat_col] == value]
        for value in target_df[cat_col].unique():
            self.target_dict[value] = target_df[target_df[cat_col] == value]
        self.top_k = top_k
        self.name = name

    def retrieve(self):
        res_l = []
        for group in self.history_dict:
            cf = ItemCF(self.history_dict[group], self.target_dict[group], self.top_k)
            res_l.append(cf.retrieve())
        res = pd.concat(res_l, ignore_index=True)
        res["method"] = "UGItemCF_" + self.name

        return res


# * ====================== User Group Retrieve Rules ====================== *


class UserGroupTimeHistory(UserGroupRetrieveRule):
    """Retrieve popular items of each **user** group in specified time window."""

    def __init__(
        self,
        data: Dict,
        customer_list: List,
        trans_df: pd.DataFrame,
        cat_cols: List,
        days: int = 7,
        n: int = 12,
        name: str = "1",
        unique: bool = True,
        item_id: str = "article_id",
        scale: bool = False,
    ):
        """Initialize TimeHistory.

        Parameters
        ----------
        data : Dict
            Data dictionary.
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        cat_cols: List
            Name of user group columns.
        n : int, optional
            Get top `n` popular items, by default ``12``.
        name : str, optional
            Name of the rule, by default ``1``.
        unique : bool, optional
            Whether to drop duplicate buying records, by default ``True``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.data = data
        self.customer_list = customer_list
        self.iid = item_id
        trans_df["t_dat"] = pd.to_datetime(trans_df["t_dat"])
        trans_df = trans_df[
            trans_df["t_dat"] > trans_df["t_dat"].max() - pd.Timedelta(days=days)
        ]
        self.trans_df = trans_df[["customer_id", self.iid, *cat_cols]]
        self.cat_cols = cat_cols
        self.unique = unique
        self.n = n
        self.name = name
        self.scale = scale

    def retrieve(self) -> List[int]:
        """Get popular items in the specified time window

        Returns:
            List[int]: top n popular items
        """
        df = self.trans_df
        if self.unique:
            df = df.drop_duplicates(["customer_id", self.iid])

        df["count"] = 1
        df = df.groupby([*self.cat_cols, self.iid], as_index=False)["count"].sum()
        df = df.sort_values(by="count", ascending=False).reset_index(drop=True)
        df = df.reset_index()

        df["rank"] = df.groupby([*self.cat_cols])["index"].rank(
            ascending=True, method="first"
        )

        if self.scale:
            df["score"] = df["count"] / df["count"].max()
        else:
            df["score"] = df["count"]
        df["method"] = "UGTimeHistory_" + self.name
        df = df[df["rank"] <= self.n][[*self.cat_cols, self.iid, "score", "method"]]

        df = self.merge(df)

        return df[["customer_id", self.iid, "method", "score"]]


class UserGroupSaleTrend(UserGroupRetrieveRule):
    """Retrieve trending items in specified time window for each user group."""

    def __init__(
        self,
        data: Dict,
        customer_list: List,
        trans_df: pd.DataFrame,
        cat_cols: List,
        days: int = 7,
        n: int = 12,
        name: str = "1",
        t: float = 0.8,
        item_id: str = "article_id",
    ):
        """Initialize SaleTrend.

        Parameters
        ----------
        data : Dict
            Data dictionary.
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        cat_cols : List
            Name of user group columns.
        days : int, optional
            Length of time window when calculating sale trend, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``12``.
        name : str, optional
            Name of the rule, by default ``1``.
        t : float, optional
            Sale trend ratio, by default ``0.8``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.data = data
        self.customer_list = customer_list
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat", *cat_cols]]
        self.cat_cols = cat_cols
        self.days = days
        self.n = n
        self.t = t
        self.name = name

    def retrieve(self):
        item_sale = self.trans_df
        item_sale["t_dat"] = pd.to_datetime(item_sale["t_dat"])
        item_sale["dat_gap"] = (item_sale.t_dat.max() - item_sale.t_dat).dt.days

        item_sale = item_sale[item_sale["dat_gap"] <= 2 * self.days - 1]
        group_a = item_sale[item_sale["dat_gap"] > self.days - 1]
        group_b = item_sale[item_sale["dat_gap"] <= self.days - 1]

        group_a["count"] = 1
        group_b["count"] = 1
        group_a = (
            group_a.groupby([*self.cat_cols, self.iid])["count"].sum().reset_index()
        )
        group_b = (
            group_b.groupby([*self.cat_cols, self.iid])["count"].sum().reset_index()
        )

        log = pd.merge(group_b, group_a, on=[*self.cat_cols, self.iid], how="left")
        log["count_x"] = log["count_x"].fillna(0)
        log["trend"] = (log["count_x"] - log["count_y"]) / (log["count_y"] + 1)

        log = log[log["trend"] > self.t]
        log = log.sort_values(by=["count_x", "trend"], ascending=False)
        log = log.reset_index(drop=True).reset_index()
        log["rank"] = log.groupby([*self.cat_cols])["index"].rank(
            ascending=True, method="first"
        )
        log = log[log["rank"] <= self.n]

        log["method"] = f"UGSaleTrend_{self.name}"
        log["score"] = log["trend"]
        log = log[[*self.cat_cols, self.iid, "method", "score"]]

        log = self.merge(log)

        return log[["customer_id", self.iid, "method", "score"]]
    




# * ======================== Global Retrieve Rules ======================== *


class TimeHistory(GlobalRetrieveRule):
    """Retrieve popular items in specified time window."""

    def __init__(
        self,
        customer_list: List,
        trans: pd.DataFrame,
        days: int=30,
        n: int = 12,
        name: str = "1",
        unique: bool = True,
        item_id: str = "article_id",
    ):
        """Initialize TimeHistory.

        Parameters
        ----------
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        n : int, optional
            Get top `n` popular items, by default ``12``.
        unique : bool, optional
            Whether to drop duplicate buying records, by default ``True``.
        days : int
            Length of time window for popularity calculation.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.customer_list = customer_list
        self.trans_df = trans[["customer_id", self.iid, "t_dat"]]
        self.unique = unique
        self.n = n
        self.name = name
        self.days = days

    def retrieve(self) -> List[int]:
        """Get popular items in the specified time window

        Returns:
            List[int]: top n popular items
        """
        df = self.trans_df.copy()
        df['t_dat'] = pd.to_datetime(df['t_dat'])

        # Lấy dữ liệu gần nhất
        max_date = df['t_dat'].max()
        min_date = max_date - pd.Timedelta(days=self.days)
        df = df[df['t_dat'] >= min_date].copy()
        if self.unique:
            df = df.drop_duplicates(["customer_id", "article_id"])

        df["count"] = 1
        df = df.groupby(self.iid, as_index=False)["count"].sum()
        df = df.sort_values(by="count", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["score"] = df["count"]
        df["method"] = "TimeHistory_" + self.name

        df = df[df["rank"] <= self.n][[self.iid, "score", "method"]]
        df = self.merge(df)

        return df[["customer_id", self.iid, "method", "score"]]


class TimeHistoryDecay(GlobalRetrieveRule):
    """Retrieve popular items in specified time window with decay."""

    def __init__(
        self,
        customer_list: List,
        trans: pd.DataFrame,
        days: int = 7,
        n: int = 12,
        name: str = "1",
        item_id: str = "article_id",
    ):
        """Initialize TimeHistoryDecay.

        Parameters
        ----------
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        days : int, optional
            Length of time window, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``12``.
        name : str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.customer_list = customer_list
        self.trans_df = trans[["customer_id", self.iid, "t_dat"]]
        self.days = days
        self.n = n
        self.name = name

    def retrieve(self):
        df = self.trans_df.copy()
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        last_ts = df["t_dat"].max()
        df["dat_gap"] = (last_ts - df["t_dat"]).dt.days

        df["last_day"] = last_ts - (last_ts - df["t_dat"]).dt.floor(f"{self.days}D")
        period_sales = (
            df[["last_day", self.iid, "t_dat"]].groupby(["last_day", self.iid]).count()
        )
        period_sales = period_sales.rename(columns={"t_dat": "period_sale"})
        df = df.join(period_sales, on=["last_day", self.iid])

        period_sales = period_sales.reset_index().set_index(self.iid)
        last_day = last_ts.strftime("%Y-%m-%d")
        df = df.join(
            period_sales.loc[period_sales["last_day"] == last_day, ["period_sale"]],
            on=self.iid,
            rsuffix="_targ",
        )
        df["period_sale_targ"].fillna(0, inplace=True)
        df["quotient"] = df["period_sale_targ"] / df["period_sale"]
        del df["period_sale_targ"], df["period_sale"]

        df["dat_gap"][df["dat_gap"] < 1] = 1
        x = df["dat_gap"]

        a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
        df["value"] = a / np.sqrt(x) + b * np.exp(-c * x) - d
        df["value"][df["value"] < 0] = 0
        df["value"] = df["value"] * df["quotient"]

        df = df.groupby([self.iid], as_index=False)["value"].sum()
        df = df.sort_values(by="value", ascending=False).reset_index(drop=True)

        df["rank"] = df.index + 1
        df["score"] = df["value"]
        df = df[df["rank"] <= self.n]
        df["method"] = f"TimeHistoryDecay_{self.name}"

        df = self.merge(df)

        return df[["customer_id", self.iid, "score", "method"]]


class SaleTrend(GlobalRetrieveRule):
    """Retrieve trending items in specified time window."""

    def __init__(
        self,
        customer_list: List,
        trans: pd.DataFrame,
        days: int = 7,
        n: int = 12,
        name: str = "1",
        t: float = 0.8,
        item_id: str = "article_id",
    ):
        """Initialize SaleTrend.

        Parameters
        ----------
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        days : int, optional
            Length of time window when calculating sale trend, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``12``.
        name : str, optional
            Name of the rule, by default ``1``.
        t : float, optional
            Sale trend ratio, by default ``0.8``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.customer_list = customer_list
        self.trans_df = trans[["customer_id", self.iid, "t_dat"]]
        self.days = days
        self.n = n
        self.t = t
        self.name = name

    def retrieve(self):
        item_sale = self.trans_df.copy()
        item_sale["t_dat"] = pd.to_datetime(item_sale["t_dat"])
        item_sale["dat_gap"] = (item_sale.t_dat.max() - item_sale.t_dat).dt.days

        item_sale = item_sale[item_sale["dat_gap"] <= 2 * self.days - 1]
        group_a = item_sale[item_sale["dat_gap"] > self.days - 1] # cũ hơn
        group_b = item_sale[item_sale["dat_gap"] <= self.days - 1] # gần đây

        group_a["count"] = 1
        group_b["count"] = 1
        group_a = group_a.groupby(self.iid)["count"].sum().reset_index()
        group_b = group_b.groupby(self.iid)["count"].sum().reset_index()

        log = pd.merge(group_b, group_a, on=[self.iid], how="left")
        log["count_y"] = log["count_y"].fillna(0)
        log["trend"] = (log["count_x"] - log["count_y"]) / log["count_x"]

        log = log[log["trend"] > self.t]
        log = log.sort_values(by=["count_x", "trend"], ascending=False)
        log = log.reset_index(drop=True).iloc[: self.n]

        log["method"] = f"SaleTrend_{self.name}"
        log["score"] = log["trend"]
        log = log[[self.iid, "method", "score"]]

        log = self.merge(log)

        return log[["customer_id", self.iid, "score", "method"]]


# * ====================== Item Group Retrieve Rule ====================== *

class ItemGroupTimeHistory(ItemGroupRetrieveRule):
    """Retrieve popular items of each **item group** in a specified time window."""

    def __init__(
        self,
        customer_list: List,
        trans: pd.DataFrame,
        item_df: pd.DataFrame,
        cat_cols: List[str],
        n: int = 12,
        name: str = "1",
        item_id: str = "article_id",
        days: int = 7,
    ):
        """
        Parameters
        ----------
        customer_list : List
            List of target customer IDs.
        trans_df : pd.DataFrame
            Transaction dataframe with at least [customer_id, article_id, t_dat].
        cat_cols : List[str]
            Columns defining item groups (e.g., ['product_type_no', 'colour_group_code']).
        n : int
            Number of top items to retrieve per group.
        name : str
            Name of the rule, for method labeling.
        item_id : str
            Column name for item ID.
        days : int
            Length of time window for popularity calculation.
        """
        self.customer_list = customer_list
        self.trans_df = trans[['customer_id', item_id, "t_dat"]]
        self.item_df = item_df[[item_id, *cat_cols]].copy()
        self.trans_df = self.trans_df.merge(
            self.item_df, on=item_id, how="left"
        )
        self.trans_df['t_dat'] = pd.to_datetime(self.trans_df['t_dat'])

        self.cat_cols = cat_cols
        self.iid = item_id
        self.n = n
        self.name = name
        self.days = days

    def retrieve(self) -> pd.DataFrame:
        df = self.trans_df  # KHÔNG copy

        # đảm bảo t_dat là datetime (nếu đã làm ở __init__ thì rất rẻ)
        t_dat = pd.to_datetime(df["t_dat"])

        max_date = t_dat.max()
        min_date = max_date - pd.Timedelta(days=self.days)

        # chỉ COPY sau khi cắt window
        df = df.loc[t_dat >= min_date].copy()
        del t_dat 

        df["count"] = 1
        df = (
            df.groupby([*self.cat_cols, self.iid], as_index=False)["count"]
            .sum()
        )

        df["rank"] = df.groupby(self.cat_cols)["count"].rank(
            ascending=False,
            method="first"
        )

        df["score"] = df["count"]
        df["method"] = f"IGTimeHistory_{self.name}"

        df = df.loc[df["rank"] <= self.n, [*self.cat_cols, self.iid, "score", "method"]]

        result_df = self.merge(df)

        result_df = (
            result_df
            .sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id")
            .head(50)
        )

        return result_df[["customer_id", self.iid, "method", "score"]]

   
class ItemGroupSaleTrend(ItemGroupRetrieveRule):
    """Retrieve trending items in a specified time window for each item group."""

    def __init__(
        self,
        customer_list: List,
        trans: pd.DataFrame,
        item_df: pd.DataFrame,
        cat_cols: List[str],
        days: int = 7,
        n: int = 12,
        name: str = "1",
        t: float = 0.8,
        item_id: str = "article_id",
    ):
        """
        Parameters
        ----------
        customer_list : List
            List of target customer IDs.
        trans_df : pd.DataFrame
            Transaction dataframe with at least ['customer_id', 'article_id', 't_dat'] + item group columns.
        cat_cols : List[str]
            Columns defining item groups (e.g., ['product_type_no', 'colour_group_code']).
        days : int
            Length of time window to calculate trends.
        n : int
            Top N items per item group.
        name : str
            Name of the rule for labeling.
        t : float
            Minimum trend ratio to consider an item trending.
        item_id : str
            Column name for item ID.
        """
        self.iid = item_id
        self.customer_list = customer_list
        self.trans_df = trans[["customer_id", self.iid, "t_dat"]]
        self.item_df = item_df[[item_id, *cat_cols]].copy()
        self.trans_df = self.trans_df.merge(
            self.item_df, on=item_id, how="left"
        )
        self.cat_cols = cat_cols
        self.days = days
        self.n = n
        self.t = t
        self.name = name

    def retrieve(self) -> pd.DataFrame:
        df = self.trans_df  # KHÔNG copy

        t_dat = pd.to_datetime(df["t_dat"])
        dat_gap = (t_dat.max() - t_dat).dt.days

        # lọc 2 * days gần nhất
        df = df.loc[dat_gap <= 2 * self.days - 1].copy()
        dat_gap = dat_gap.loc[df.index]

        group_a = df.loc[dat_gap > self.days - 1].copy()   # cũ hơn
        group_b = df.loc[dat_gap <= self.days - 1].copy()  # gần đây

        del t_dat 
        del dat_gap
        group_a["count"] = 1
        group_b["count"] = 1

        group_a = group_a.groupby([*self.cat_cols, self.iid])["count"].sum().reset_index()
        group_b = group_b.groupby([*self.cat_cols, self.iid])["count"].sum().reset_index()

        log = pd.merge(group_b, group_a, on=[*self.cat_cols, self.iid], how="left")
        log["count_y"] = log["count_y"].fillna(0)
        log["trend"] = (log["count_x"] - log["count_y"]) / log["count_x"]

        log = log[log["trend"] > self.t]
        log["rank"] = log.groupby(self.cat_cols)["trend"].rank(
            ascending=False, method="first"
        )
        log = log[log["rank"] <= self.n]

        log["method"] = f"ItemGroupSaleTrend_{self.name}"
        log["score"] = log["trend"]

        log = log[[*self.cat_cols, self.iid, "method", "score"]]

        result_df = self.merge(log)
        result_df = (
            result_df
            .sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id")
            .head(50)
        )

        return result_df[["customer_id", self.iid, "method", "score"]]

class ItemSimilarity(PersonalRetrieveRule):
    """Retrieve similar items based on precomputed embeddings or TF-IDF."""

    def __init__(
        self,
        customer_list: List,
        trans_df: pd.DataFrame,
        text_sim_map: dict = None,
        days: int = 7,
        name: str = "1",
        item_id: str = "article_id",
        top_k: int = 50
    ):
        """
        Parameters
        ----------
        articles_df : pd.DataFrame
            Dataframe chứa item info.
        embedding_dict : dict
            Precomputed embeddings {article_id: embedding_vector}.
        text_sim_map : dict
            Precomputed similarity map {article_id: [(similar_item, score), ...]}.
        top_k : int
            Số item gợi ý tối đa cho mỗi item.
        """
        self.customer_list = customer_list
        self.trans_df = trans_df.copy()
        self.text_sim_map = text_sim_map      # {article_id: [(item, score), ...]}
        self.top_k = top_k
        self.days = days
        self.name = name
        self.iid = item_id

    def retrieve(self) -> pd.DataFrame:
        """
        Parameters
        ----------
        customer_trans : pd.DataFrame
            Transactions of customers with ['customer_id', 'article_id', 't_dat'].
        start_date, end_date : datetime, optional
            Chỉ lấy transactions trong khoảng thời gian này.
        """
        df = self.trans_df.copy()
        df['t_dat'] = pd.to_datetime(df['t_dat'])

        df = df[df['t_dat'] >= df['t_dat'].max() - pd.Timedelta(days=self.days)]

        # Đếm số lần mỗi item được mua trong khoảng thời gian
        
        purchased_items = df.groupby('customer_id')['article_id'].apply(set).to_dict()

        results = []
        for customer_id in self.customer_list:
            if customer_id not in purchased_items:
                continue
            items = purchased_items[customer_id]
            item_sim_sub = {}
            for item in items:
                for sim_item, score in self.text_sim_map.get(item, []):
                    item_sim_sub[sim_item] = item_sim_sub.get(sim_item, 0) + score
            
            for sim_item, score in item_sim_sub.items():
                res_df = {
                    'customer_id': customer_id,
                    'article_id': sim_item,
                    'score': score,
                }
                
            results.append(res_df)

        df_result = pd.DataFrame(results)
        # Sắp xếp: item vừa mua gần nhất, số lần mua cao -> ưu tiên
        df_result = df_result.sort_values(by='score', ascending=False)
        df_result = df_result.groupby('customer_id').head(self.top_k).reset_index(drop=True)
        df_result['method'] = 'ItemSimilarity_' + self.name
        
        return df_result
    
    
import numpy as np
import pandas as pd
from typing import List, Optional
from abc import ABC
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class ItemGroupRAG(ItemGroupRetrieveRule):
    """
    Retrieve top-K items from each item group based on embedding similarity (RAG-style).
    
    Requires:
        - precomputed item embeddings
        - mapping from embedding index to article_id
    """

    def __init__(
        self,
        item_embedding_file: str,
        top_k: int = 12,
        group_col: Optional[str] = None,
        method_name: str = "ItemGroupRAG",
    ):
        """
        Parameters
        ----------
        item_embedding_file : str
            Path to pickle file containing {"item_ids": ..., "embeddings": ...}
        top_k : int
            Number of items to retrieve
        group_col : str, optional
            If items are grouped (like product_type), provide group column
        method_name : str
            Name of this rule
        """
        with open(item_embedding_file, "rb") as f:
            store = pickle.load(f)
        self.item_ids = store["item_ids"]  # np.array or list
        self.embeddings = store["embeddings"]  # np.array (num_items x dim)
        self.top_k = top_k
        self.group_col = group_col
        self.method_name = method_name

        # Optional: build mapping if group_col exists
        self.item_group_map = None
        if self.group_col and "group_cols" in store:
            # store["group_cols"] should map index -> group value
            self.item_group_map = store["group_cols"]

    def retrieve(self, query_vectors: np.ndarray, query_groups: Optional[List] = None) -> pd.DataFrame:
        """
        Retrieve top-K items for each query vector (user vector or user profile).

        Parameters
        ----------
        query_vectors : np.ndarray
            Array of shape (num_queries, embedding_dim)
        query_groups : List, optional
            Group of each query. If provided, restrict retrieval to items in same group

        Returns
        -------
        pd.DataFrame
            Columns: ['group_col' (optional), 'article_id', 'method', 'score']
        """
        all_rows = []

        for i, qvec in enumerate(query_vectors):
            qvec = qvec.reshape(1, -1)  # 1 x dim

            if self.group_col and query_groups is not None:
                group = query_groups[i]
                # mask items in the same group
                mask = np.array(self.item_group_map) == group
                emb_subset = self.embeddings[mask]
                ids_subset = np.array(self.item_ids)[mask]
            else:
                emb_subset = self.embeddings
                ids_subset = np.array(self.item_ids)

            # Cosine similarity
            scores = cosine_similarity(qvec, emb_subset)[0]
            topk_idx = np.argsort(scores)[::-1][:self.top_k]

            retrieved_ids = ids_subset[topk_idx]
            retrieved_scores = scores[topk_idx]

            row_dict = {
                "article_id": retrieved_ids,
                "score": retrieved_scores,
            }
            if self.group_col and query_groups is not None:
                row_dict[self.group_col] = [query_groups[i]] * self.top_k

            row_dict["method"] = [self.method_name] * self.top_k

            # Convert to DataFrame
            df_row = pd.DataFrame(row_dict)
            all_rows.append(df_row)

        result_df = pd.concat(all_rows, ignore_index=True)
        return result_df

# import cudf
# import cupy as cp
# from cuml.neighbors import NearestNeighbors
# from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix


# class UserToUserClusterCandidate(PersonalRetrieveRule):
#     """User-to-User CF using item clusters for similarity."""

#     def __init__(
#         self,
#         customer_list: List[str],
#         history_df: pd.DataFrame,
#         articles_df: pd.DataFrame,
#         cat_cols: str,
#         article_id: str = 'article_id',
#         days_window: int = 30,
#         k_neighbors: int = 3,
#         n_candidates: int = 100
#     ):
#         self.cat_cols = cat_cols
#         self.days_window = days_window
#         self.k_neighbors = k_neighbors
#         self.n_candidates = n_candidates
#         self.iid = article_id

#         # Chuyển sang GPU
#         self.history_df = cudf.from_pandas(history_df) if isinstance(history_df, pd.DataFrame) else history_df
#         self.articles_df = cudf.from_pandas(articles_df) if isinstance(articles_df, pd.DataFrame) else articles_df
#         self.customer_list = customer_list
#         self.output_columns = ['customer_id', 'article_id', 'method', 'score']

#     def retrieve(self) -> pd.DataFrame:
#         # 1️⃣ Lọc transaction theo thời gian
#         last_date = self.history_df['t_dat'].max()
#         start_date = last_date - pd.Timedelta(days=self.days_window)
#         recent_df = self.history_df[self.history_df['t_dat'] >= start_date]

#         # 2️⃣ Merge với cluster_col
#         clusters = self.articles_df[[self.iid, self.cat_cols]]
#         recent = recent_df.merge(clusters, on=self.iid, how='left')
#         if len(recent) == 0:
#             return pd.DataFrame(columns=self.output_columns)

#         # 3️⃣ Factorize user và cluster
#         recent['user_code'], user_index = recent['customer_id'].factorize()
#         recent['cluster_code'], cluster_index = recent[self.cat_cols].factorize()

#         unique_interactions = recent[['user_code', 'cluster_code']].drop_duplicates()

#         # 4️⃣ Build sparse User x Cluster matrix
#         rows = unique_interactions['user_code'].values
#         cols = unique_interactions['cluster_code'].values
#         data = cp.ones(len(rows), dtype=cp.float32)
#         n_users = len(user_index)
#         n_clusters = len(cluster_index)
#         user_cluster_matrix = cp_csr_matrix((data, (rows, cols)), shape=(n_users, n_clusters))

#         # 5️⃣ Fit KNN
#         k_actual = min(self.k_neighbors + 1, n_users)
#         if k_actual <= 1:
#             return pd.DataFrame(columns=self.output_columns)

#         knn_model = NearestNeighbors(n_neighbors=k_actual, metric='cosine', algorithm='brute')
#         knn_model.fit(user_cluster_matrix)

#         # 6️⃣ Lọc target users
#         target_users_cudf = cudf.Series(self.customer_list)
#         user_map = cudf.DataFrame({'customer_id': user_index, 'code': cp.arange(len(user_index))})
#         target_indices = target_users_cudf.to_frame('customer_id').merge(user_map, on='customer_id', how='inner')['code']

#         if len(target_indices) == 0:
#             return pd.DataFrame(columns=self.output_columns)

#         target_matrix = user_cluster_matrix[target_indices.values.get(), :]
#         distances, indices = knn_model.kneighbors(target_matrix, n_neighbors=k_actual)

#         # 7️⃣ Tạo candidates từ neighbor
#         neighbor_indices = indices[:, 1:]  # loại chính user
#         n_targets = len(target_indices)
#         k_real = neighbor_indices.shape[1]

#         neighbors_flat = neighbor_indices.flatten()
#         targets_repeated = cp.repeat(target_indices.values, k_real)

#         df_neighbors = cudf.DataFrame({
#             'user_code': targets_repeated,
#             'neighbor_code': neighbors_flat
#         })

#         history_subset = recent[['user_code', self.iid]].drop_duplicates()

#         candidates = df_neighbors.merge(
#             history_subset,
#             left_on='neighbor_code',
#             right_on='user_code',
#             how='inner'
#         ).rename(columns={self.iid: 'article_id', 'user_code_x': 'user_code'})

#         # 8️⃣ Anti-join: loại bỏ item user đã mua
#         target_history = recent[['user_code', self.iid]].drop_duplicates()
#         candidates = candidates.merge(
#             target_history,
#             on=['user_code', 'article_id'],
#             how='leftanti'
#         )

#         # 9️⃣ Tính score và lấy top-N
#         scored = candidates.groupby(['user_code', 'article_id']).size().reset_index(name='count')
#         scored['score'] = scored['count'] / self.k_neighbors
#         scored = scored.sort_values(['user_code', 'score'], ascending=[True, False])
#         top_candidates = scored.groupby('user_code').head(self.n_candidates)

#         # 10️⃣ Map lại user_code → customer_id
#         final = top_candidates.merge(user_map, left_on='user_code', right_on='code', how='left')
#         final['method'] = f'u2u_{self.cluster_col}'
#         final = final[['customer_id', 'article_id', 'method', 'score']]

#         return final.to_pandas()
        

# * ============================ Filter Rules ============================ *


class OutOfStock(FilterRule):
    """Filter items that are out of stock."""

    def __init__(self, trans_df: pd.DataFrame, item_id: str = "article_id"):
        """Initialize OutOfStock.

        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        item_id : str, optional
           Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        mask = trans_df["t_dat"] >= "2020-08-01"
        self.trans_df = trans_df.loc[mask, ["customer_id", self.iid, "t_dat"]]

    def _off_stock_items(self) -> List[int]:
        """Get items that are no longer for sale

        Returns:
            List: list of off stock items
        """
        sale = self.trans_df
        sale["t_dat"] = pd.to_datetime(sale["t_dat"])
        sale["year_month"] = (
            (sale["t_dat"].dt.year).astype(str)
            + "_"
            + (sale["t_dat"].dt.month).astype(str)
        )
        sale = (
            sale.groupby([self.iid, "year_month"])["customer_id"]
            .count()
            .reset_index(name="count")
        )

        sale = pd.pivot_table(
            sale, values="count", index=self.iid, columns="year_month"
        )
        sale = sale.fillna(0)
        mask = ((sale["2020_9"] - sale["2020_8"]) / sale["2020_8"]) < -0.8
        mask2 = sale["2020_9"] == 0

        return list(sale[mask | mask2].index)

    def retrieve(self) -> List:
        off_stock = self._off_stock_items()

        return off_stock
