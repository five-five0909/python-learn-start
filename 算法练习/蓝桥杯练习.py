from typing import List
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 使用三指针【2首1尾】
        # 做一个字典
        n=len(nums)
        if n<3:
            return []
        sorted_arr=sorted(nums)
        result_arr=[]
        for i in range(n-2):
            if sorted_arr[i]>0:
                break
            # 去重
            if i>0 and sorted_arr[i]==sorted_arr[i-1]:
                continue
            j,k=i+1,n-1
            while j<k:
                if sorted_arr[i] + sorted_arr[j] + sorted_arr[k] == 0:
                    # 成立，装入数组
                    result_arr.append([sorted_arr[i] , sorted_arr[j] , sorted_arr[k]])
                    # 去重
                    while j<k and sorted_arr[j]==sorted_arr[j+1]:
                        j+=1
                    while j<k and sorted_arr[k]==sorted_arr[k-1]:
                        k-=1
                    # 移动双指针以寻找新的解
                    j += 1
                    k -= 1
                elif sorted_arr[i] + sorted_arr[j] + sorted_arr[k] > 0:
                    k-=1
                elif sorted_arr[i] + sorted_arr[j] + sorted_arr[k] < 0:
                    j+=1
        return result_arr
    
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals)<2:
            return intervals
        # 先把数组进行排序，按照第一个位置进行排序
        intervals.sort(key=lambda x : x[0])
        # 初始化结果列表
        merged_arr=[intervals[0]]
        for i in range(1,len(intervals)):
            # -1表示取末尾
            last_merged_arr=merged_arr[-1]
            cur_to_merged=intervals[i]
            # 检查是否重叠
            if cur_to_merged[0] <= last_merged_arr[1]:
                last_merged_arr[1]=max(last_merged_arr[1],cur_to_merged[1])
            else:
                merged_arr.append(cur_to_merged)
        return merged_arr