import math


class TransferExecutor(object):
    def execute(self, network, plans):
        """
        目的：把原来分散在 network.execute_energy_transfer 和 PowerControlScheduler.execute_plans 的能量执行逻辑，统一放到一个执行器内，规避主循环里的分支判断。

        统一执行计划：支持直连与多跳；优先使用 plan['energy_sent']，否则回退 donor.E_char。
        保持与原 network.execute_energy_transfer/PowerControlScheduler.execute_plans 行为一致。
        """
        for plan in plans:
            donor = plan.get("donor")
            receiver = plan.get("receiver")
            path = plan.get("path") or []
            distance = float(plan.get("distance", 0.0))

            if donor is None or receiver is None or len(path) < 2:
                continue

            # 下发能量：优先计划内指定量
            energy_sent = plan.get("energy_sent", getattr(donor, "E_char", 0.0))
            try:
                energy_sent = float(energy_sent)
            except Exception:
                energy_sent = float(getattr(donor, "E_char", 0.0))

            if distance <= math.sqrt(3):
                # 近距离直接传输
                eta = donor.energy_transfer_efficiency(receiver)
                energy_received = energy_sent * eta

                donor.current_energy -= donor.energy_consumption(receiver, transfer_WET=True)
                receiver.current_energy += energy_received

                donor.transferred_history.append(energy_sent)
                receiver.received_history.append(energy_received)
            else:
                # 多跳：逐跳衰减
                energy_left = energy_sent
                donor.transferred_history.append(energy_sent)
                for i in range(len(path) - 1):
                    sender = path[i]
                    recv_i = path[i + 1]
                    eta = sender.energy_transfer_efficiency(recv_i)
                    delivered = energy_left * eta

                    transfer_WET = (i == 0)
                    sender.current_energy -= sender.energy_consumption(recv_i, transfer_WET=transfer_WET)

                    recv_i.current_energy += delivered
                    recv_i.received_history.append(delivered)

                    energy_left = delivered


